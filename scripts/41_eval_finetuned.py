"""Evaluate a LoRA-fine-tuned MedGemma adapter on the 754 test studies and score
13-class F1 against the zero-shot baseline (same scorer, same reference).

Uses the SAME detection instruction the model was trained on (scripts/40), applies
the adapter via mlx_vlm.load(adapter_path=...), parses the model's semicolon list of
canonical finding names with normalize_dx, and writes a predictions file the existing
eval machinery reads. Run AFTER training (MLX is single-GPU / not thread-safe).

  python3 scripts/41_eval_finetuned.py --adapter results/adapters/medgemma_det
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src.eval_accuracy import (_pred_set, _truth_positive_set, propagate_hierarchy,
                               RELIABLE_CLASSES)
from src.grouping import CORE_PATHOLOGY_GROUPS, to_groups, any_acute
from src.labeling import CHEXPERT_CLASSES
from src.normalize import normalize_dx

FINDABLE = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
# MUST match scripts/40_finetune_medgemma.INSTRUCTION exactly (train/eval parity).
INSTRUCTION = (
    "You are a radiologist reading a chest radiograph. Decide which of the following "
    "findings are present: " + ", ".join(FINDABLE) + ". "
    "Reply with only the present findings by name, separated by semicolons, or "
    "'No acute cardiopulmonary finding' if none are present."
)
ALL = FINDABLE


def micro(yt, yp):
    yt, yp = np.array(yt, int), np.array(yp, int)
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def run_inference(adapter, model_id, uids, studies, limit):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load(model_id, adapter_path=adapter,
                            processor_config={"trust_remote_code": True})
    config = model.config
    rows = []
    for i, uid in enumerate(uids):
        r = studies.loc[uid]
        paths = [p for p in (r["frontal_paths"] or []) if p and os.path.exists(p)][:1]
        messages = [{"role": "user", "content": INSTRUCTION}]
        prompt = apply_chat_template(processor, config, messages, num_images=len(paths))
        out = generate(model, processor, prompt, image=paths or None,
                       max_tokens=128, temperature=0.0, verbose=False)
        raw = out.text if hasattr(out, "text") else str(out)
        pos = {c for c in normalize_dx(raw) if c != "No Finding"}
        findings = {c: {"presence": "present", "confidence": 80} for c in pos}
        primary = sorted(pos)[0] if pos else "No acute cardiopulmonary finding"
        rows.append({"uid": uid, "prompt_condition": "medgemma_ft",
                     "response": json.dumps({"primary_diagnosis": primary,
                                             "findings": findings, "raw_response": raw}),
                     "primary_diagnosis": primary, "schema_valid": True})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(uids)}", flush=True)
    return pd.DataFrame(rows)


def score(pred_df, studies, label):
    g = pred_df.set_index("uid")
    uids = [u for u in g.index if u in studies.index]
    T = {u: propagate_hierarchy(_truth_positive_set(studies.loc[u], "ones", "lblcx_")) for u in uids}
    P = {u: propagate_hierarchy(_pred_set(g.loc[u])) for u in uids}
    f13 = micro([[1 if x in T[u] else 0 for x in ALL] for u in uids],
                [[1 if x in P[u] else 0 for x in ALL] for u in uids])
    frel = micro([[1 if x in T[u] else 0 for x in RELIABLE_CLASSES] for u in uids],
                 [[1 if x in P[u] else 0 for x in RELIABLE_CLASSES] for u in uids])
    k = list(CORE_PATHOLOGY_GROUPS)
    f3 = micro([[1 if x in to_groups(T[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in uids],
               [[1 if x in to_groups(P[u], CORE_PATHOLOGY_GROUPS) else 0 for x in k] for u in uids])
    fps = np.mean([len(P[u]) for u in uids])
    print(f"{label:<26} n={len(uids)}  13cls F1={f13[2]:.3f} (P={f13[0]:.3f} R={f13[1]:.3f})  "
          f"reliable={frel[2]:.3f}  3grp={f3[2]:.3f}  find/std={fps:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="results/adapters/medgemma_det")
    ap.add_argument("--model", default="mlx-community/medgemma-4b-it-8bit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-suffix", default="medgemma_ft")
    args = ap.parse_args()
    cfg = load_cfg("configs/config_medgemma_estr.yaml")
    out = cfg["paths"]["out_dir"]
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    studies["frontal_paths"] = studies["frontal_paths"].map(
        lambda x: eval(x) if isinstance(x, str) else x)
    test = pd.read_csv(os.path.join(out, "test_manifest.csv"))
    uids = [u for u in test.uid if u in studies.index]
    if args.limit:
        uids = uids[:args.limit]
    print(f"Fine-tuned inference on {len(uids)} test studies (adapter={args.adapter})...")
    pred = run_inference(args.adapter, args.model, uids, studies, args.limit)
    pred.to_csv(os.path.join(out, f"predictions_test_{args.out_suffix}.csv"), index=False)
    print(f"Wrote predictions_test_{args.out_suffix}.csv\n")
    # comparison
    print("=== 13-class comparison (CheXbert ref, hierarchy-aware) ===")
    score(pred, studies, "MedGemma FINE-TUNED")
    base = pd.read_csv(os.path.join(out, "predictions_test_medgemma.csv"))
    for c in ["direct", "evidence_first"]:
        score(base[base.prompt_condition == c], studies, f"MedGemma zero-shot ({c})")


if __name__ == "__main__":
    main()
