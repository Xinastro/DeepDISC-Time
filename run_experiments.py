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
    p.add_argument("--experiment", choices=["a", "a-sub", "b", "b1", "b2", "c", "d"], required=True)
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
    p.add_argument("--flux-fractions", default="0.25,0.5,1.0,2.0,4.0",
                   help="Experiment a-sub: mean nuclear flux in units of the "
                        "nominal detection threshold. Values below 1.0 are the "
                        "sub-threshold regime where the detection claim lives.")
    p.add_argument("--gradients", default="0,0.25,0.5,0.75,1.0",
                   help="Experiment D: bulge-minus-disc colour gradients, magnitudes")
    p.add_argument("--n-scenes", type=int, default=40,
                   help="Experiment D: scenes generated per gradient value")
    p.add_argument("--with-coadd", action="store_true",
                   help="Experiment A: add the true inverse-variance coadd arm. "
                        "The --static-model arm is mean FEATURE pooling, which is "
                        "not detection on a coadd; without this flag the result "
                        "cannot be described as beating a static coadd detector.")
    p.add_argument("--match-method", default="greedy_score",
                   choices=["greedy_score", "hungarian"],
                   help="source matching rule. Report both in severe blends: a "
                        "conclusion that survives only one is a conclusion about "
                        "the matcher.")
    p.add_argument("--timing-repeats", type=int, default=5,
                   help="Experiment B: timed repeats, median reported")
    p.add_argument("--timing-warmup", type=int, default=2,
                   help="Experiment B: discarded warm-up iterations")
    p.add_argument("--split", default=None,
                   help="split manifest JSON from scripts/make_split.py. Strongly "
                        "recommended: without it the experiment runs on whatever "
                        "sequences.pkl contains, which for injected data usually "
                        "means the same hosts and visits appeared in training.")
    p.add_argument("--split-part", default="test",
                   help="which part of the split to evaluate on")
    p.add_argument("--supervision", default="injection",
                   choices=["injection", "scarlet2", "mixed", "none"],
                   help="where the evaluated model's training labels came from. "
                        "Decides eligibility for an independent-accuracy claim: a "
                        "model distilled from scarlet2 can answer B1 but not B2.")
    p.add_argument("--manifest-out", default=None,
                   help="write a RunManifest JSON (environment, versions, split "
                        "digest) next to the results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.experiment in ("b1", "b2"):
        # B1 and B2 are different questions and must not be run as one.
        # B1 asks whether an amortized model reproduces scarlet2-TD solutions
        # faster; the reference is the teacher's output and training on scarlet2
        # labels is fair. B2 asks which method recovers injection truth; a model
        # distilled from scarlet2 in the same domain is ineligible.
        from deepdisc_time.eval.experiments import (
            experiment_b1_distillation,
            experiment_b2_truth_accuracy,
        )

        temporal = model_backend(
            load_model(args.model, "attn_var", args.device, role="temporal"), args.device
        )
        os.makedirs(args.out, exist_ok=True)

        if args.experiment == "b1":
            if not args.with_scarlet2:
                raise SystemExit(
                    "B1 is a comparison against scarlet2-TD solutions, so it "
                    "requires --with-scarlet2. Note that the scarlet2 adapter "
                    "has never been executed: run "
                    "tests/test_scarlet2_integration.py first."
                )
            from deepdisc_time.baselines import Scarlet2TDBackend

            df = experiment_b1_distillation(
                seqs,
                temporal,
                lambda x: Scarlet2TDBackend().fit_many(x),
                n_repeats=args.timing_repeats,
                n_warmup=args.timing_warmup,
                device=args.device,
            )
            out = os.path.join(args.out, "experiment_b1_distillation.csv")
        else:
            backends = {"temporal": temporal}
            if args.static_model:
                backends["static"] = model_backend(
                    load_model(args.static_model, "mean", args.device, role="static null"),
                    args.device,
                )
            if args.with_coadd:
                from deepdisc_time.baselines import CoaddDetectionBackend

                backends["coadd"] = CoaddDetectionBackend().fit_many
            if args.with_scarlet2:
                from deepdisc_time.baselines import Scarlet2TDBackend

                backends["scarlet2_td"] = lambda x: Scarlet2TDBackend().fit_many(x)
            df = experiment_b2_truth_accuracy(
                seqs, backends, supervision=args.supervision
            )
            out = os.path.join(args.out, "experiment_b2_truth_accuracy.csv")

        df.to_csv(out, index=False)
        print(df.to_string(index=False))
        print(f"wrote {out} ({len(df)} rows)")
        return

    if args.experiment == "a-sub":
        # The sub-threshold detection sweep: the measurement the detection claim
        # needs. Generates its own scenes, because the point is a controlled
        # sweep of mean nuclear flux across the detection threshold at fixed
        # variability amplitude, which a stored dataset does not provide.
        from deepdisc_time.eval.experiments import experiment_a_sub_threshold

        temporal = model_backend(
            load_model(args.model, "attn_var", args.device, role="temporal"), args.device
        )
        backends = {"temporal": temporal}
        if args.static_model:
            backends["static"] = model_backend(
                load_model(args.static_model, "mean", args.device, role="static null"),
                args.device,
            )
        if args.with_coadd:
            from deepdisc_time.baselines import CoaddDetectionBackend

            backends["coadd"] = CoaddDetectionBackend().fit_many
        fr = tuple(float(x) for x in args.flux_fractions.split(","))
        df = experiment_a_sub_threshold(
            backends, flux_fractions=fr, n_scenes=args.n_scenes
        )
        os.makedirs(args.out, exist_ok=True)
        out = os.path.join(args.out, "experiment_a_sub_threshold.csv")
        df.to_csv(out, index=False)
        print(df.to_string(index=False))
        print(f"wrote {out} ({len(df)} rows)")
        print(
            "\nRead this as completeness against mean flux, one line per arm. The "
            "claim lives in the rows with sub_threshold=True: if the temporal and "
            "static curves coincide there, the result is about deblending and "
            "photometry of candidate detections, not about detection."
        )
        return

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

    split_digest = split_strategy = ""
    if args.split:
        from deepdisc_time.data.splits import SplitManifest, audit_split

        man = SplitManifest.load(args.split)
        split_digest, split_strategy = man.digest(), man.strategy
        parts = man.split(seqs, strict=False)
        rep = audit_split(man, seqs, pair=("train", args.split_part))
        print(f"split {man.name!r} ({man.strategy}) digest {man.digest()}")
        for w in rep["warnings"]:
            print(f"  LEAKAGE WARNING: {w}")
        if not rep["warnings"]:
            print("  audit: clean on every channel measured")
        seqs = parts.get(args.split_part, [])
        if not seqs:
            raise SystemExit(
                f"split part {args.split_part!r} is empty. Parts available: "
                f"{sorted(parts)}"
            )
        print(f"evaluating on the {args.split_part!r} part")
    else:
        print(
            "WARNING: no --split given. For injected data this usually means the "
            "same hosts, backgrounds and visits appeared in training, which "
            "inflates every number below. Build one with scripts/make_split.py."
        )

    if args.limit:
        seqs = seqs[: args.limit]
    print(f"{len(seqs)} sequences")

    if args.manifest_out:
        from deepdisc_time.eval.provenance import RunManifest

        RunManifest(
            run_name=os.path.basename(args.out.rstrip("/")) or "run",
            supervision=args.supervision,
            experiment=args.experiment,
            split_digest=split_digest,
            split_strategy=split_strategy,
            config=vars(args),
        ).save(args.manifest_out)
        print(f"wrote run manifest {args.manifest_out}")

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
        coadd = None
        if args.with_coadd:
            from deepdisc_time.baselines import CoaddDetectionBackend

            cb = CoaddDetectionBackend()
            coadd = cb.fit_many
        recs = experiment_a_detection(seqs, temporal, static, coadd_backend=coadd)
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

        # Synchronised, warmed-up, repeated timing.  The per-scene wall clock
        # recorded inside SceneSolution is adequate for a CPU smoke test and
        # wrong on a GPU: CUDA kernels are asynchronous, so an unsynchronised
        # timer measures enqueue time, and the error is larger for the faster
        # method -- the direction that flatters an amortized model.  Quote the
        # numbers from this file, not from the solution objects.
        from deepdisc_time.eval.timing import benchmark_backends

        timing = benchmark_backends(
            backends,
            seqs,
            n_repeats=args.timing_repeats,
            n_warmup=args.timing_warmup,
            device=args.device,
        )
        write_csv(timing, os.path.join(args.out, "experiment_b_timing.csv"))
        return

    recs = experiment_c_epoch_ablation(seqs, temporal, static)
    write_csv(recs, os.path.join(args.out, "experiment_c_epoch_ablation.csv"))


if __name__ == "__main__":
    main()
