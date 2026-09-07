"""Build the labeled study table, stratify, and draw a balanced test set.

Outputs (results/):
  studies_labeled.csv   all usable studies with labels + difficulty
  test_manifest.csv     locked balanced test set (uid list + stratum + hash)
  dev_manifest.csv      held-out dev set for prompt/rubric/judge tuning
"""
import hashlib
import os

import pandas as pd
from _bootstrap import ROOT, load_cfg

from src.data import Paths, load_studies
from src.labeling import label_frame
from src.difficulty import stratify


def main():
    cfg = load_cfg()
    seed = cfg["seed"]
    p = cfg["paths"]
    print("Loading + joining IU X-Ray ...")
    df = load_studies(Paths(p["reports_csv"], p["proj_csv"], p["images_dir"]),
                      require_findings=cfg["dataset"]["require_findings"])
    print(f"  usable studies: {len(df)}")

    print("Labeling reports (rule-based CheXpert-style) ...")
    df = label_frame(df)                                    # lbl_*  from findings+impression
    df = label_frame(df, text_col="impression",            # lblimp_* = acute bottom line
                     prefix="lblimp_", summary=False)
    print("Assigning difficulty strata ...")
    df = stratify(df)

    out = p["out_dir"]
    df.to_csv(os.path.join(out, "studies_labeled.csv"), index=False)
    print("\nDifficulty distribution (all usable studies):")
    print(df["difficulty"].value_counts().to_string())

    # Balanced test set: sample up to target per stratum; rest eligible for dev.
    targets = cfg["dataset"]["strata_targets"]
    test_parts, used = [], set()
    for stratum, n in targets.items():
        pool = df[df.difficulty == stratum]
        take = pool.sample(min(n, len(pool)), random_state=seed)
        test_parts.append(take)
        used.update(take.uid.tolist())
    test = pd.concat(test_parts).reset_index(drop=True)

    # Dev set: fraction of the remaining (non-test) studies, stratified.
    remain = df[~df.uid.isin(used)]
    dev = (remain.groupby("difficulty", group_keys=False)[remain.columns]
           .apply(lambda g: g.sample(max(1, int(len(g) * cfg["dataset"]["dev_fraction"])),
                                     random_state=seed), include_groups=True))

    def manifest(sub):
        m = sub[["uid", "difficulty", "n_positive", "n_uncertain",
                 "is_hedged", "xxxx_density"]].copy()
        m["report_hash"] = sub["report_text"].map(
            lambda t: hashlib.md5(t.encode()).hexdigest()[:12])
        return m

    manifest(test).to_csv(os.path.join(out, "test_manifest.csv"), index=False)
    manifest(dev).to_csv(os.path.join(out, "dev_manifest.csv"), index=False)
    print(f"\nLocked TEST set: {len(test)} studies")
    print(test["difficulty"].value_counts().to_string())
    print(f"\nDEV set: {len(dev)} studies")
    print(f"\nWrote manifests to {out}")


if __name__ == "__main__":
    main()
