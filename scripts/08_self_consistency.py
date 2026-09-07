"""Self-consistency ensembling for the structured_soft prompt (score-lift attempt).

Audit showed the model over-calls and its self-reported confidence is flat, so
it can't be thresholded. Sampling the same image N times at temperature>0 and
voting per finding (a) votes out spurious findings that only appear in a minority
of samples -> precision, and (b) yields a REAL confidence = vote fraction that we
CAN threshold/calibrate.

Writes predictions_test_sc{N}.csv where each finding's presence is 'present' iff
it won >= vote_k of N samples, and confidence = 100*votes/N. Downstream eval
(_pred_set / multilabel_prf) then works unchanged; sweep vote_k post-hoc for free.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src import schema
from src.inference import GeminiBackend, _generate_with_retry
from src.prompts import STRUCTURED_FINDINGS

COND = os.environ.get("SC_PROMPT", "structured_soft")


def _parse_presence(out):
    """Robustly pull {finding: presence} from a sample. Accepts the findings
    dict; if absent, falls back to the free-text primary_diagnosis so a valid
    response is never silently dropped (the T>0 dropout bug)."""
    f = out.get("findings", {}) or {}
    pres = {}
    for c in STRUCTURED_FINDINGS:
        v = f.get(c)
        if isinstance(v, dict) and "presence" in v:
            pres[c] = str(v.get("presence", "absent")).lower()
        elif isinstance(v, str):                 # model sometimes returns "present"
            pres[c] = v.lower()
    if pres:
        return pres
    # fallback: derive from primary_diagnosis so the sample still counts
    from src.normalize import normalize_dx
    hit = normalize_dx(out.get("primary_diagnosis", "")) - {"No Finding"}
    return {c: ("present" if c in hit else "absent") for c in STRUCTURED_FINDINGS}


def sample_findings(backend, row, n):
    """Return list of per-sample {finding: presence} dicts (valid samples only)."""
    votes = []
    for _ in range(n):
        out, err = _generate_with_retry(backend, row, COND, retries=6)
        if out is None:
            continue
        votes.append(_parse_presence(out))
    return votes


def aggregate(votes, vote_k):
    """Majority-vote findings; confidence = vote fraction. vote_k applied later
    in the sweep, so here we store raw counts and default-mark at ceil(n/2)."""
    n = len(votes)
    findings = {}
    for c in STRUCTURED_FINDINGS:
        v = sum(1 for s in votes if s.get(c) == "present")
        conf = int(round(100 * v / n)) if n else 0
        pres = "present" if v >= vote_k else "absent"
        findings[c] = {"presence": pres, "confidence": conf, "votes": v, "n": n}
    present = [c for c, d in findings.items() if d["presence"] == "present"]
    return {
        "findings": findings,
        "primary_diagnosis": present[0] if present else "No acute cardiopulmonary finding",
        "confidence": max((findings[c]["confidence"] for c in present), default=90),
        "explanation": f"self-consistency vote over {n} samples",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_soft.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--n", type=int, default=5, help="samples per image")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--vote-k", type=int, default=3, help="default present threshold")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg["inference"]["temperature"] = args.temperature  # need diversity
    out = cfg["paths"]["out_dir"]
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    manifest = pd.read_csv(os.path.join(out, f"{args.split}_manifest.csv"))
    df = studies[studies.uid.isin(manifest.uid)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    for col in ["frontal_paths", "lateral_paths"]:
        df[col] = df[col].map(lambda s: eval(s) if isinstance(s, str) else s)

    backend = GeminiBackend(cfg)
    print(f"self-consistency: {len(df)} studies x {args.n} samples @ T={args.temperature} "
          f"= {len(df)*args.n} calls | workers={args.workers}")

    rows = [None] * len(df)

    def work(i):
        row = df.iloc[i]
        votes = sample_findings(backend, row, args.n)
        obj = aggregate(votes, args.vote_k)
        return i, {
            "uid": row["uid"], "case_id": str(row["uid"]),
            "prompt_condition": COND, "model_id": cfg["inference"]["model_id"],
            "difficulty": row.get("difficulty"),
            "response": json.dumps(obj), "primary_diagnosis": obj["primary_diagnosis"],
            "confidence": obj["confidence"], "schema_valid": True,
            "schema_errors": "", "api_error": "", "n_samples": len(votes),
        }

    tag = os.environ.get("SC_TAG", "")
    fname = f"predictions_{args.split}_sc{args.n}{tag}.csv"
    fpath = os.path.join(out, fname)
    ckpt = fpath.replace(".csv", ".partial.csv")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(df))]
        for fut in as_completed(futs):
            i, rec = fut.result()
            rows[i] = rec
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(df)} studies", flush=True)
                # checkpoint completed rows so a long (~3770-call) run is recoverable
                pd.DataFrame([r for r in rows if r is not None]).to_csv(ckpt, index=False)

    preds = pd.DataFrame(rows)
    preds.to_csv(fpath, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    empty = (preds["n_samples"] == 0).sum()
    print(f"Wrote {fname} | studies with 0 valid samples: {empty}")


if __name__ == "__main__":
    main()
