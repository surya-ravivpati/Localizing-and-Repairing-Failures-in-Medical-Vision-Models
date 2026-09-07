"""Build the expert-labeled validation manifest (protocol §3.4 upgrade).

Two goals drive the design:
  G1 (unbiased ceiling): a representative random sample -> true F1 vs a human
      reading the IMAGE, free of report-incompleteness bias.
  G2 (resolve over-calling): an enriched sample of DISPUTED over-calls
      (model-present / reference-absent) -> measure what fraction are genuine
      false positives vs findings that are visible but were never dictated.

Output: results/expert_manifest.csv — one row per study, blinded (no reference /
model labels in the columns the labeler sees), with a stable shuffle order and a
hidden `_stratum` tag for later analysis. Image paths included for the tool.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import _pred_set, _truth_positive_set, propagate_hierarchy
from src.labeling import CHEXPERT_CLASSES

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
# classes whose over-calling dominates the precision loss (from the 754 audit)
HIGH_FP = ["Atelectasis", "Support Devices", "Lung Opacity", "Cardiomegaly"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=120, help="representative stratum")
    ap.add_argument("--n-disputed", type=int, default=80, help="over-call stratum")
    ap.add_argument("--seed", type=int, default=20260717)
    args = ap.parse_args()

    cfg = load_cfg()
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    tg = pd.read_csv(os.path.join(out, "predictions_test_targeted.csv")).set_index("uid")
    uids = sorted(set(tg.index) & set(s.index))
    rng = np.random.default_rng(args.seed)

    # per-study: model positives, reference positives, disputed-FP classes
    disputed = {}
    for u in uids:
        t = propagate_hierarchy(_truth_positive_set(s.loc[u], "ignore", "lblcx_"))
        p = propagate_hierarchy(_pred_set(tg.loc[u]))
        disputed[u] = (p - t)   # model-present, reference-absent

    # STRATUM A: representative random, proportional across difficulty
    diff = s.loc[uids, "difficulty"]
    a_uids = []
    for stratum, grp in diff.groupby(diff):
        k = max(1, round(args.n_random * len(grp) / len(uids)))
        a_uids += list(rng.choice(grp.index.values, size=min(k, len(grp)), replace=False))
    a_uids = a_uids[:args.n_random]

    # STRATUM B: enriched disputed over-calls, spread across the high-FP classes
    remaining = [u for u in uids if u not in set(a_uids)]
    b_uids, per_class = [], max(1, args.n_disputed // len(HIGH_FP))
    for c in HIGH_FP:
        cands = [u for u in remaining
                 if c in disputed[u] and u not in set(b_uids)]
        pick = rng.choice(cands, size=min(per_class, len(cands)), replace=False)
        b_uids += list(pick)
    # top up to n_disputed with any remaining disputed studies
    extra = [u for u in remaining if disputed[u] and u not in set(b_uids)]
    rng.shuffle(extra)
    b_uids += extra[:max(0, args.n_disputed - len(b_uids))]

    rows = []
    for u, stratum in [(u, "random") for u in a_uids] + [(u, "disputed") for u in b_uids]:
        fr = s.loc[u, "frontal_paths"]
        fr = eval(fr) if isinstance(fr, str) else fr
        lat = s.loc[u, "lateral_paths"]
        lat = eval(lat) if isinstance(lat, str) else lat
        rows.append({
            "uid": u,
            "frontal_path": fr[0] if fr else "",
            "lateral_path": lat[0] if lat else "",
            "indication": s.loc[u, "indication"],
            "_stratum": stratum,                        # hidden from labeler
            "_model_positives": ";".join(sorted(_pred_set(tg.loc[u]))),   # for analysis
            "_disputed_classes": ";".join(sorted(disputed[u])),           # for analysis
            **{f"label_{c}": "" for c in ALL},          # blank cols the expert fills
            "label_No Finding": "",
        })
    df = pd.DataFrame(rows).drop_duplicates("uid")
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # blind order
    path = os.path.join(out, "expert_manifest.csv")
    df.to_csv(path, index=False)

    print(f"expert set: {len(df)} studies "
          f"({(df._stratum=='random').sum()} random + {(df._stratum=='disputed').sum()} disputed)")
    print(f"  random stratum difficulty:\n"
          f"{s.loc[df[df._stratum=='random'].uid,'difficulty'].value_counts().to_string()}")
    tot_disp = sum(len(d.split(';')) for d in df._disputed_classes if d)
    print(f"  total disputed over-call cells to adjudicate: {tot_disp}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
