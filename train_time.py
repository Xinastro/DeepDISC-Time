#!/usr/bin/env python
"""Train DeepDISC-Time.

Two paths.  Without ``--config-file`` this trains the pure-torch
``TemporalSceneModel``, which runs anywhere and is the right thing for
development and for the amortization timing harness.  With ``--config-file``
it hands off to the Detectron2 LazyConfig path, which is the production route.

Examples
--------
    python scripts/train_time.py --data data/dev --max-iter 2000
    python scripts/train_time.py --data data/dev --pool-mode mean   # Experiment A null
    python scripts/train_time.py --config-file src/deepdisc_time/configs/deepdisc_time_ddf.py \\
        --num-gpus 4 --run-name ddf_v1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import torch
from torch.utils.data import DataLoader

from deepdisc_time.data.loaders import EpochSampler, SequenceDataset, collate_sequences
from deepdisc_time.modeling.meta_arch import TemporalSceneModel
from deepdisc_time.training.trainers import (
    MaskedEpochPretrainer,
    TemporalTrainer,
    TrainConfig,
    warm_start_from_static,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="directory containing sequences.pkl")
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--config-file", help="Detectron2 LazyConfig; switches to the production path")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--run-name", default="deepdisc_time")
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=32, help="sequence length cap")
    p.add_argument("--epoch-strategy", default="uniform",
                   choices=["first", "uniform", "random", "contiguous"])
    # Architecture ablation knobs. "mean" is the Experiment A null: the closest
    # feature-space analogue of detecting on a coadd.
    p.add_argument("--pool-mode", default="attn_var", choices=["mean", "max", "attn", "attn_var"])
    p.add_argument("--flux-head", default="flow", choices=["flow", "mdn"])
    p.add_argument("--no-predict-sed", action="store_true",
                   help="disable the per-band static amplitude (v0.1.0 behaviour). "
                        "This is the control arm of Experiment D: it is also the "
                        "assumption scarlet2 makes, so the with/without comparison "
                        "measures the cost of a shared assumption rather than a "
                        "difference between packages.")
    p.add_argument("--sigma-model-frac", type=float, default=0.0,
                   help="model-error floor as a fraction of model flux, added in "
                        "quadrature to the pixel variance. The variance plane "
                        "describes photon and read noise only; it says nothing "
                        "about background residuals, PSF-model error or host-model "
                        "inadequacy, and without a floor the rendering term "
                        "becomes a confident teacher of the wrong decomposition "
                        "in the brightest pixels. Sweep it and report the "
                        "sensitivity rather than trusting the default.")
    p.add_argument("--split", default=None,
                   help="split manifest JSON from scripts/make_split.py; trains on "
                        "the 'train' part only. Without it, training uses every "
                        "sequence in the directory, which for injected data means "
                        "the test hosts were seen.")
    p.add_argument("--split-part", default="train")
    p.add_argument("--no-posterior-quantiles", action="store_true",
                   help="skip exporting per-epoch posterior quantiles at "
                        "inference (saves space, costs honest calibration)")
    p.add_argument("--lambda-render", type=float, default=1.0,
                   help="0 ablates the physics-informed rendering loss")
    p.add_argument("--warm-start", help="static DeepDISC checkpoint to band-average into the stem")
    p.add_argument("--pretrain-iter", type=int, default=0,
                   help="masked-epoch self-supervised iterations before supervised training")
    p.add_argument("--supervision", default="injection",
                   choices=["injection", "scarlet2", "mixed", "none"],
                   help="where this run's labels come from. Recorded in the "
                        "checkpoint so that downstream experiments can check "
                        "eligibility for an independent-accuracy claim rather "
                        "than relying on anyone remembering.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config_file:
        # Production path: defer entirely to the DeepDISC launcher so that the
        # distributed setup, hooks and checkpointing behave identically to the
        # group's static-imaging runs.
        from detectron2.engine import launch
        from deepdisc.utils.parse_arguments import make_training_arg_parser  # noqa: F401

        raise SystemExit(
            "Detectron2 path: run the DeepDISC launcher with this config, e.g.\n"
            f"  python -m deepdisc.scripts.run_model --cfgfile {args.config_file} "
            f"--num-gpus {args.num_gpus} --run-name {args.run_name}\n"
            "This wrapper exists so the two paths share one entry point; the "
            "production launcher is not reimplemented here."
        )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.data, "sequences.pkl"), "rb") as f:
        seqs = pickle.load(f)
    print(f"loaded {len(seqs)} sequences")

    if args.split:
        from deepdisc_time.data.splits import SplitManifest

        man = SplitManifest.load(args.split)
        parts = man.split(seqs, strict=False)
        seqs = parts.get(args.split_part, [])
        if not seqs:
            raise SystemExit(
                f"split part {args.split_part!r} is empty; parts: {sorted(parts)}"
            )
        print(
            f"split {man.name!r} ({man.strategy}) digest {man.digest()}: "
            f"training on {len(seqs)} sequences from the {args.split_part!r} part"
        )
    else:
        print(
            "WARNING: no --split given, so training uses every sequence in the "
            "directory. For injected data that means the test hosts and visits "
            "were seen during training. Build a split with scripts/make_split.py."
        )

    dataset = SequenceDataset(
        sequences=seqs,
        sampler=EpochSampler(max_epochs=args.max_epochs, strategy=args.epoch_strategy),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
        drop_last=True,
    )

    model = TemporalSceneModel(
        pool_mode=args.pool_mode,
        flux_head=args.flux_head,
        lambda_render=args.lambda_render,
        predict_sed=not args.no_predict_sed,
        posterior_quantiles=not args.no_posterior_quantiles,
    )
    if args.sigma_model_frac > 0:
        from deepdisc_time.modeling.rendering import RenderingLoss

        model.rendering_loss = RenderingLoss(sigma_model_frac=args.sigma_model_frac)
    if args.warm_start:
        report = warm_start_from_static(model, args.warm_start)
        print(f"warm start: {len(report['loaded'])} loaded, {len(report['skipped'])} skipped")
        for line in report["skipped"][:10]:
            print(f"  skipped: {line}")

    cfg = TrainConfig(lr=args.lr, max_iter=args.max_iter, device=args.device)

    if args.pretrain_iter:
        print(f"masked-epoch pretraining for {args.pretrain_iter} iterations")
        MaskedEpochPretrainer(model.to(args.device), loader, config=cfg).train(args.pretrain_iter)

    trainer = TemporalTrainer(model, loader, cfg)
    history = trainer.train()

    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(args.out, "model.pth"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f)
    print(f"wrote {args.out}/model.pth")


if __name__ == "__main__":
    main()
