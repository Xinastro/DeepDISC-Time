#!/usr/bin/env python
"""Build, audit and freeze a train/val/test split.

Run this once per dataset and commit the resulting JSON. A split that exists
only as a line of code in a notebook is a split nobody can check, and "we used a
spatial split" is not a statement a referee can verify.

Examples
--------
Field-level holdout, which is the one to use for a headline result::

    python scripts/make_split.py --data data/ddf_all --strategy field \\
        --out splits/field_holdout.json

Note that ``--strategy field+base`` is weaker than ``field`` on the visit
channel, not stronger, because composing groupers makes them finer and allows
hosts from one field onto both sides. The audit will tell you; the name will
not.

Spatial blocks composed with injection-variant grouping, for development::

    python scripts/make_split.py --data data/ecdfs_injected \\
        --strategy sky+base --block-deg 0.1 --out splits/ecdfs_dev.json

The audit runs automatically and prints every leakage channel it can measure.
A non-zero visit overlap is expected for a spatial split inside one field and
disqualifying for a headline result; the script reports and does not decide.
Use --require-clean in a pipeline to make it decide.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

from deepdisc_time.data.splits import (
    audit_split,
    build_manifest,
    by_base_scene,
    by_field,
    by_night,
    by_sky_block,
    compose_groupers,
)


def build_grouper(name: str, block_deg: float):
    if name == "field":
        return by_field, "by_field"
    if name == "sky":
        return by_sky_block(block_deg), f"by_sky_block({block_deg})"
    if name == "night":
        return by_night, "by_night"
    if name == "base":
        return by_base_scene(), "by_base_scene"
    if name == "sky+base":
        return (
            compose_groupers(by_sky_block(block_deg), by_base_scene()),
            f"by_sky_block({block_deg}) + by_base_scene",
        )
    if name == "field+base":
        return (
            compose_groupers(by_field, by_base_scene()),
            "by_field + by_base_scene",
        )
    raise SystemExit(f"unknown strategy {name!r}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="directory containing sequences.pkl")
    p.add_argument("--out", required=True, help="where to write the manifest JSON")
    p.add_argument(
        "--strategy",
        default="field",
        choices=["field", "sky", "night", "base", "sky+base", "field+base"],
        help="grouping. 'field' is the strongest available and the one for a "
        "headline result: it holds whole fields out, so visits, sky and PSF "
        "patterns are all disjoint. 'sky' alone does NOT close visit-level "
        "leakage. Note that 'field+base' is WEAKER than 'field' on the visit "
        "channel, because composing groupers makes groups finer and lets hosts "
        "from one field land on both sides; see the splits module docstring. "
        "Whatever you choose, read the audit rather than trusting the name.",
    )
    p.add_argument("--block-deg", type=float, default=0.25)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--notes", default="")
    p.add_argument(
        "--require-clean",
        action="store_true",
        help="exit non-zero if the audit reports any warning. Use this in a "
        "pipeline that must not proceed on a leaky split.",
    )
    args = p.parse_args()

    with open(os.path.join(args.data, "sequences.pkl"), "rb") as f:
        seqs = pickle.load(f)
    print(f"{len(seqs)} sequences from {args.data}")

    grouper, label = build_grouper(args.strategy, args.block_deg)
    fractions = {"train": args.train, "val": args.val, "test": args.test}
    fractions = {k: v for k, v in fractions.items() if v > 0}

    m = build_manifest(
        seqs,
        grouper,
        fractions,
        seed=args.seed,
        name=args.name or os.path.splitext(os.path.basename(args.out))[0],
        strategy=label,
        notes=args.notes,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    m.save(args.out)
    print(f"wrote {args.out}")
    print(f"  strategy : {m.strategy}")
    print(f"  digest   : {m.digest()}   <- quote this in logs and captions")
    print(f"  groups   : {len(set(m.groups.values()))}")
    print(f"  fractions: { {k: round(v, 3) for k, v in m.fractions.items()} }")

    pairs = [("train", "test")]
    if "val" in fractions:
        pairs.append(("train", "val"))

    any_warn = False
    for pair in pairs:
        rep = audit_split(m, seqs, pair=pair)
        print(f"\n--- audit {pair[0]} vs {pair[1]} ---")
        print(f"  group overlap        : {len(rep['group_overlap'])}  (must be 0)")
        print(f"  base-scene overlap   : {len(rep['base_scene_overlap'])}")
        print(
            f"  visit overlap        : {len(rep['visit_overlap'])} "
            f"({rep['visit_overlap_frac']:.0%} of visits)"
        )
        print(f"  night overlap        : {len(rep['night_overlap'])}")
        sep = rep["min_separation_arcsec"]
        print(
            "  min separation       : "
            + ("inf" if sep == float("inf") else f"{sep:.1f} arcsec")
        )
        print(f"  counts               : {rep['counts']}")
        if rep["warnings"]:
            any_warn = True
            print("  WARNINGS:")
            for w in rep["warnings"]:
                print(f"    - {w}")
        else:
            print("  clean on every channel measured")

        side = os.path.splitext(args.out)[0] + f".audit_{pair[0]}_{pair[1]}.json"
        with open(side, "w") as f:
            json.dump(rep, f, indent=2, default=str)
        print(f"  audit written to {side}")

    if any_warn and args.require_clean:
        print(
            "\n--require-clean was set and the audit reported warnings; exiting 1.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
