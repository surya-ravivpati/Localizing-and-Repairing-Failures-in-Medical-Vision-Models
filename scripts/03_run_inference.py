"""Run all 5 prompt conditions over the locked test set (protocol §5-6).

Backend is set in config (mock | anthropic). Writes results/predictions.csv.
"""
import argparse
import os

import pandas as pd
from _bootstrap import load_cfg

from src.inference import run_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--limit", type=int, default=None, help="cap #studies (debug)")
    ap.add_argument("--config", default=None, help="path to a config yaml")
    ap.add_argument("--out-suffix", default=None,
                    help="tag for output file, e.g. gemini -> predictions_test_gemini.csv")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out = cfg["paths"]["out_dir"]
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    manifest = pd.read_csv(os.path.join(out, f"{args.split}_manifest.csv"))
    df = studies[studies.uid.isin(manifest.uid)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    # frontal/lateral path columns were stringified on save; parse back to lists
    for col in ["frontal_paths", "lateral_paths"]:
        df[col] = df[col].map(lambda s: eval(s) if isinstance(s, str) else s)

    print(f"Backend={cfg['inference']['backend']} model={cfg['inference']['model_id']} "
          f"| {len(df)} studies x {len(cfg['inference']['prompts'])} prompts "
          f"| workers={cfg['inference'].get('max_workers', 8)}")
    tag = f"_{args.out_suffix}" if args.out_suffix else ""
    ckpt = os.path.join(out, f"predictions_{args.split}{tag}.partial.csv")
    preds = run_inference(df, cfg, checkpoint=ckpt)
    valid = preds.schema_valid.mean()
    n_err = (preds["api_error"] != "").sum() if "api_error" in preds else 0
    print(f"Generated {len(preds)} responses | schema-valid: {valid*100:.1f}% | "
          f"api errors: {n_err}")
    tag = f"_{args.out_suffix}" if args.out_suffix else ""
    fname = f"predictions_{args.split}{tag}.csv"
    preds.to_csv(os.path.join(out, fname), index=False)
    print(f"Wrote {fname}")


if __name__ == "__main__":
    main()
