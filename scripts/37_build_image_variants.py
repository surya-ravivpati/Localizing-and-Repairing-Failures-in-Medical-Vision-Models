"""Render Experiment 1 image variants to disk (one condition at a time).

Experiment 1 varies ONLY the visual input. Every arm's image parts are rendered
here first, then inference runs with `image_max_side: 0` so all arms take the
byte-identical passthrough path (see src/imaging.py for the full rationale).

Writes:
  results/exp1_images/{variant}/{uid}_{kind}.png   the parts themselves
  results/exp1_image_manifest_{variant}.csv        provenance for every part
  results/exp1_images/_qa/{variant}_contact.png    labelled grid for eyeball QA

Pilot first, then full:
  python3 scripts/37_build_image_variants.py --variant baseline --limit 24
  python3 scripts/37_build_image_variants.py --variant baseline --limit 0
"""
import argparse
import os

import pandas as pd
from _bootstrap import load_cfg

from src import imaging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_exp1_gemini.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", default="baseline",
                    choices=list(imaging.IMPLEMENTED_VARIANTS))
    ap.add_argument("--limit", type=int, default=24,
                    help="pilot-first; 0 = the whole split")
    ap.add_argument("--out-root", default=None,
                    help="default: {out_dir}/exp1_images")
    ap.add_argument("--qa-studies", type=int, default=6)
    ap.add_argument("--uids-file", default=None,
                    help="CSV with a 'uid' column: restrict to exactly these studies "
                         "(Experiment 2 uses one seeded stratified subset for every rung)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out = cfg["paths"]["out_dir"]
    out_root = args.out_root or os.path.join(out, "exp1_images")

    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    manifest = pd.read_csv(os.path.join(out, f"{args.split}_manifest.csv"))
    df = studies[studies.uid.isin(manifest.uid)].reset_index(drop=True)
    if args.uids_file:
        keep = pd.read_csv(args.uids_file).uid
        df = df[df.uid.isin(keep)].reset_index(drop=True)
        print(f"restricted to {len(df)} studies from {args.uids_file}", flush=True)
    if args.limit:
        df = df.head(args.limit)
    for col in ["frontal_paths", "lateral_paths"]:
        df[col] = df[col].map(lambda s: eval(s) if isinstance(s, str) else s)

    print(f"Rendering variant={args.variant} for {len(df)} studies -> {out_root}",
          flush=True)

    # Arm D: resolve one zone per study BEFORE rendering. Two passes are required --
    # the marginal used to assign zones to normals/non-localizable studies can only
    # be built once every genuine zone is known.
    zmap = {}
    if args.variant == "wrongcrop_f":
        zmap = imaging.assign_wrong_zones(
            list(zip(df.uid, df.get("mesh", pd.Series(index=df.index, dtype=object)))),
            cfg.get("seed", 20260705))
        df = df[df.uid.isin(zmap)].reset_index(drop=True)
        print(f"  wrong-crop control: {len(zmap)} studies with a true zone to "
              f"deliberately mis-point", flush=True)
    elif imaging.VARIANT_SPECS[args.variant]["crops"] == "oracle":
        zmap = imaging.assign_oracle_zones(
            list(zip(df.uid, df.get("mesh", pd.Series(index=df.index, dtype=object)))),
            cfg.get("seed", 20260705))
        loc = sum(1 for v in zmap.values() if v[1] == "localizable")
        print(f"  oracle zones: {loc}/{len(zmap)} localizable from MeSH, "
              f"{len(zmap)-loc} assigned from the matched marginal", flush=True)

    parts = []
    for i, row in df.iterrows():
        z, zsrc = zmap.get(row["uid"], (None, ""))
        parts.extend(imaging.render_variant(row, args.variant, out_root,
                                            zone=z, zone_source=zsrc))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)}", flush=True)

    mdf = pd.DataFrame(parts, columns=imaging.MANIFEST_COLUMNS)
    mp = imaging.manifest_path(out, args.variant)
    mdf.to_csv(mp, index=False)

    per_study = mdf.groupby("uid").size()
    print(f"\nWrote {len(mdf)} parts for {mdf.uid.nunique()} studies -> {mp}")
    print(f"  parts/study: {dict(per_study.value_counts().sort_index())}")
    print(f"  kinds:       {dict(mdf.kind.value_counts())}")
    if "zone_source" in mdf and (mdf.zone_source != "").any():
        zs = mdf[mdf.zone_source != ""]
        print(f"  zone_source: {dict(zs.zone_source.value_counts())}")
        print(f"  zone distribution by source:")
        for src_, g in zs.groupby("zone_source"):
            frac = (g.zone.value_counts(normalize=True).round(3)).to_dict()
            print(f"    {src_:<12} {frac}")
    print(f"  max side px: min={mdf[['width','height']].max(axis=1).min()} "
          f"median={int(mdf[['width','height']].max(axis=1).median())} "
          f"max={mdf[['width','height']].max(axis=1).max()}")
    dupes = len(mdf) - mdf.sha256.nunique()
    print(f"  duplicate-byte parts: {dupes}")

    sheet = imaging.contact_sheet(parts, os.path.join(out_root, "_qa",
                                                      f"{args.variant}_contact.png"),
                                  max_studies=args.qa_studies)
    if sheet:
        print(f"\nQA contact sheet -> {sheet}\n  OPEN IT and confirm the images "
              f"before spending any API budget.")


if __name__ == "__main__":
    main()
