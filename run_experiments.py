#!/usr/bin/env python
"""Run Experiments A, B and C and write tidy CSV records.

Analysis and presentation stay separate: this writes records, not plots.

Examples
--------
    python scripts/run_experiments.py --data data/dev --experiment a --model outputs/run/model.pth
    python scripts/run_experiments.py --data data/dev --experiment b --with-scarlet2
    python scripts/run_experiments.py --data data/dev --experiment c
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle

import torch

from deepdisc_time.baselines import DIAForcedPhotometry, StaticThenForcedPhotometry
from deepdisc_time.data.loaders import sequences_to_batch
from deepdisc_time.eval.experiments import (
    experiment_a_detection,
    experiment_b_amortization,
    experiment_c_epoch_ablation,
)
from deepdisc_time.modeling.meta_arch import TemporalSceneModel


def load_model(path, expected_pool_mode, device, role=""):
    """Load a checkpoint, refusing to silently mismatch the architecture.

    Experiments A and C compare a temporally aggregating model against a null
    that pools epochs by mean.  Those are different architectures with
    different parameters, so they must be *trained separately*.  Loading an
    ``attn_var`` checkpoint into a ``mean`` model would compare a trained model
    against a partially initialised one, and the resulting "temporal helps"
    conclusion would be an artefact of the load, not a result.

    This function therefore reads the ``pool_mode`` recorded in the checkpoint
    and raises on a mismatch rather than falling back to ``strict=False``.
    """
    if not path or not os.path.exists(path):
        print(
            f"warning: no checkpoint at {path!r} for the {role or 'model'} arm; "
            "using an untrained model. Numbers from this run are a plumbing "
            "check, not a result."
        )
        return TemporalSceneModel(pool_mode=expected_pool_mode).to(device).eval()

    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved = (ckpt.get("args") or {}).get("pool_mode")
    if saved is not None and saved != expected_pool_mode:
        raise SystemExit(
            f"checkpoint {path} was trained with pool_mode={saved!r}, but the "
            f"{role or 'requested'} arm needs pool_mode={expected_pool_mode!r}.\n"
            "Train the null separately, then pass it explicitly:\n"
            f"  python scripts/train_time.py --data <data> --pool-mode {expected_pool_mode} "
            "--out outputs/null\n"
            "  python scripts/run_experiments.py ... --static-model outputs/null/model.pth\n"
            "Loading across architectures would compare a trained model against "
            "a partially initialised one."
        )

    # Every architecture-shaping flag must be read back from the checkpoint,
    # not assumed.  Reconstructing a model with different flags and loading
    # anyway would silently compare a trained network against a partially
    # initialised one, which is how a spurious result gets published.
    # ``predict_sed`` defaults to False for checkpoints written before v0.2.0,
    # which have no ``sed_decoder`` in their state dict.
    saved_args = ckpt.get("args") or {}
    predict_sed = not saved_args.get("no_predict_sed", False)
    if "no_predict_sed" not in saved_args:
        predict_sed = any(k.startswith("roi_head.sed_decoder") for k in ckpt["model"])
    model = TemporalSceneModel(
        pool_mode=expected_pool_mode,
        flux_head=saved_args.get("flux_head", "flow"),
        predict_sed=predict_sed,
    )
    model.load_state_dict(ckpt["model"])
    if role:
        print(f"loaded {role}: pool_mode={expected_pool_mode}, predict_sed={predict_sed}")
    return model.to(device).eval()


def model_backend(model, device, batch_size=4):
    """Wrap a model as a ``sequences -> solutions`` callable.

    Batched deliberately: the amortization argument is about throughput, and
    timing an amortized model one scene at a time would understate it as badly
    as timing a per-scene optimiser on a whole batch would overstate it.
    """

    def _run(seqs):
        out = []
        for i in range(0, len(seqs), batch_size):
            chunk = list(seqs[i : i + batch_size])
            batch = sequences_to_batch(chunk, device=device, with_truth=False)
            out.extend(model.predict_scene(batch, [s.scene_id for s in chunk]))
        return out

    return _run


def write_csv(records, path):
    if not records:
        print("no records")
        return
    keys = sorted({k for r in records for k in r})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"wrote {path} ({len(records)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--experiment", choices=["a", "b", "c", "d"], required=True)
    p.add_argument("--model", default="outputs/run/model.pth")
    p.add_argument("--static-model", default=None,
                   help="checkpoint trained with --pool-mode mean (the Experiment A null)")
    p.add_argument("--out", default="outputs/experiments")
    p.add_argument("--with-scarlet2", action="store_true",
                   help="include the scarlet2-TD reference baseline (slow, needs scarlet2)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--nosed-model", default=None,
                   help="checkpoint trained with --no-predict-sed; adds the control "
                        "arm to Experiment D")
    p.add_argument("--gradients", default="0,0.25,0.5,0.75,1.0",
                   help="Experiment D: bulge-minus-disc colour gradients, magnitudes")
    p.add_argument("--n-scenes", type=int, default=40,
                   help="Experiment D: scenes generated per gradient value")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.experiment == "d":
        # Experiment D generates its own scenes: the nucleus must be exactly
        # constant and the host gradient exactly controlled, neither of which a
        # stored dataset provides.  --data is ignored here on purpose.
        from deepdisc_time.eval.experiments import experiment_d_colour_gradient

        temporal = model_backend(
            load_model(args.model, "attn_var", args.device, role="temporal"), args.device
        )
        backends = {"deepdisc_time": temporal}
        if args.nosed_model:
            backends["deepdisc_time_nosed"] = model_backend(
                load_model(args.nosed_model, "attn_var", args.device, role="no-SED control"),
                args.device,
            )
        else:
            print(
                "note: running without --nosed-model. The with/without per-band "
                "amplitude comparison is the point of this experiment; train a "
                "control arm with --no-predict-sed and re-run."
            )
        if args.with_scarlet2:
            from deepdisc_time.baselines import Scarlet2TDBackend

            backends["scarlet2_td"] = lambda s: Scarlet2TDBackend().fit_many(s)
        grads = tuple(float(x) for x in args.gradients.split(","))
        df = experiment_d_colour_gradient(
            backends, gradients=grads, n_scenes=args.n_scenes
        )
        os.makedirs(args.out, exist_ok=True)
        out = os.path.join(args.out, "experiment_d_colour_gradient.csv")
        df.to_csv(out, index=False)
        print(df.to_string(index=False))
        print(f"wrote {out} ({len(df)} rows)")
        return

    with open(os.path.join(args.data, "sequences.pkl"), "rb") as f:
        seqs = pickle.load(f)
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"{len(seqs)} sequences")

    temporal = model_backend(
        load_model(args.model, "attn_var", args.device, role="temporal"), args.device
    )
    static = None
    if args.experiment in ("a", "c"):
        if not args.static_model:
            raise SystemExit(
                "Experiments A and C need a separately trained null.\n"
                "  python scripts/train_time.py --data <data> --pool-mode mean --out outputs/null\n"
                "then re-run with --static-model outputs/null/model.pth"
            )
        static = model_backend(
            load_model(args.static_model, "mean", args.device, role="static null"), args.device
        )

    if args.experiment == "a":
        recs = experiment_a_detection(seqs, temporal, static)
        write_csv(recs, os.path.join(args.out, "experiment_a_detection.csv"))
        return

    if args.experiment == "b":
        backends = {
            "deepdisc_time": temporal,
            "static_then_forced": lambda s: StaticThenForcedPhotometry().fit_many(s),
            "dia_forced": lambda s: DIAForcedPhotometry().fit_many(s),
        }
        if args.with_scarlet2:
            from deepdisc_time.baselines import Scarlet2TDBackend

            backends["scarlet2_td"] = lambda s: Scarlet2TDBackend().fit_many(s)
        recs = experiment_b_amortization(seqs, backends)
        write_csv(recs, os.path.join(args.out, "experiment_b_amortization.csv"))
        return

    recs = experiment_c_epoch_ablation(seqs, temporal, static)
    write_csv(recs, os.path.join(args.out, "experiment_c_epoch_ablation.csv"))


if __name__ == "__main__":
    main()
