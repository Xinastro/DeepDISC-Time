#!/usr/bin/env python
"""Build a multi-epoch sequence dataset and cache the visit index.

The index is the expensive part and the reusable part.  Building it once and
writing it to disk avoids a registry query per scene per epoch at training
time, which is the difference between an IO-bound job and an unusable one.

Examples
--------
Synthetic grid for development::

    python scripts/make_sequences.py --mode synthetic --n 512 --out data/dev

DP2 via the Butler (run on the Rubin Science Platform)::

    python scripts/make_sequences.py --mode butler \\
        --repo /repo/dp2 --collections LSSTCam/runs/DRP/DP2 \\
        --where "instrument='LSSTCam' AND visit.target_name='ECDFS'" \\
        --positions ecdfs_agn.csv --out data/ecdfs
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from deepdisc_time.data.injection import InjectionConfig, make_synthetic_sequence


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["synthetic", "fits", "butler"], default="synthetic")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--n", type=int, default=256, help="number of scenes (synthetic)")
    p.add_argument("--stamp-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--bands", default=None, help="comma-separated band filter, e.g. g,r,i")
    p.add_argument("--quality-max", type=int, default=1,
                   help="1 keeps below-coadd-threshold DP2 visits, which the model down-weights")
    p.add_argument("--seed", type=int, default=0)
    # synthetic grid
    p.add_argument("--nucleus-to-host", default="0.05,0.1,0.2,0.5,1.0")
    p.add_argument("--neighbour-sep", default="none,4,8,16")
    # real data
    p.add_argument("--pattern", help="FITS glob (fits mode)")
    p.add_argument("--repo", help="Butler repo (butler mode)")
    p.add_argument("--collections", help="comma-separated collections")
    p.add_argument("--where", default="", help="registry query")
    p.add_argument("--positions", help="CSV with ra,dec columns")
    return p.parse_args()


def build_synthetic(args):
    """Sweep the blending/variability grid from the proposal."""
    rng = np.random.default_rng(args.seed)
    n2h = [float(v) for v in args.nucleus_to_host.split(",")]
    seps = [None if v == "none" else float(v) for v in args.neighbour_sep.split(",")]

    seqs = []
    for i in range(args.n):
        cfg = InjectionConfig(
            seed=int(rng.integers(0, 2**31)),
            nucleus_to_host=float(rng.choice(n2h)),
            neighbour_separation=seps[int(rng.integers(len(seps)))],
            neighbour_flux_ratio=float(rng.uniform(0.2, 3.0)),
            host_half_light_radius=float(rng.uniform(1.5, 6.0)),
            host_sersic_n=float(rng.choice([1.0, 2.0, 4.0])),
            host_ellipticity=float(rng.uniform(0.0, 0.5)),
            host_pa=float(rng.uniform(0, np.pi)),
            # Non-zero nuclear offsets exercise the nuclear/non-nuclear call,
            # which is a headline scarlet2-TD application and therefore a
            # capability the amortized model has to match, not skip.
            nuclear_offset=float(rng.choice([0.0, 0.0, 0.0, 1.5, 3.0])),
            drw_tau=float(10 ** rng.uniform(1.5, 3.0)),
            drw_sf_inf=float(10 ** rng.uniform(-1.3, -0.2)),
        )
        seqs.append(
            make_synthetic_sequence(
                cfg,
                n_epochs=args.max_epochs or 32,
                size=args.stamp_size,
                bands=tuple(args.bands.split(",")) if args.bands else ("g", "r", "i"),
            )
        )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n}", flush=True)
    return seqs


def build_real(args):
    from deepdisc_time.data.stamps import ButlerStampService, FitsStampService

    kw = dict(
        stamp_size=args.stamp_size,
        max_epochs=args.max_epochs,
        band_filter=args.bands.split(",") if args.bands else None,
        quality_max=args.quality_max,
    )
    if args.mode == "fits":
        svc = FitsStampService(**kw)
        n = svc.build_index(args.pattern)
    else:
        svc = ButlerStampService(args.repo, args.collections.split(","), **kw)
        n = svc.build_index(where=args.where)
    print(f"indexed {n} visits")

    positions = np.genfromtxt(args.positions, delimiter=",", names=True)
    seqs = []
    for i, row in enumerate(positions):
        try:
            seqs.append(svc.get_sequence(float(row["ra"]), float(row["dec"]), f"scene_{i:07d}"))
        except ValueError as exc:
            print(f"  skipping scene {i}: {exc}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(positions)}", flush=True)
    return seqs, svc.index


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    index = None
    if args.mode == "synthetic":
        seqs = build_synthetic(args)
    else:
        seqs, index = build_real(args)

    with open(os.path.join(args.out, "sequences.pkl"), "wb") as f:
        pickle.dump(seqs, f, protocol=4)
    if index is not None:
        with open(os.path.join(args.out, "visit_index.pkl"), "wb") as f:
            pickle.dump(index, f, protocol=4)

    meta = {
        "mode": args.mode,
        "n_scenes": len(seqs),
        "stamp_size": args.stamp_size,
        "epochs_median": float(np.median([s.n_epochs for s in seqs])) if seqs else 0.0,
        "epochs_min": int(min(s.n_epochs for s in seqs)) if seqs else 0,
        "epochs_max": int(max(s.n_epochs for s in seqs)) if seqs else 0,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
