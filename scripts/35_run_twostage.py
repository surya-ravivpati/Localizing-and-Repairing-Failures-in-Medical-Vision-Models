"""Two-stage perception/reasoning separation (reviewer follow-up to EDIT 1).

evidence_structured generates the diagnosis in the SAME pass as the findings, so a
wrong final answer can't be attributed to bad perception vs bad reasoning. This
splits it into two INDEPENDENT calls, run sequentially per study:

    Stage A (perception_only): image -> findings ONLY, diagnosis forbidden.
    Stage B (diagnose_from_findings): the FROZEN Stage-A findings, as text, with
        NO image access -> diagnosis only.

Stage A reuses the existing single-call `run_inference` machinery unchanged (it's
just another condition). Stage B is a new sequential text-only pass — no backend
duplication; it calls the same backend's new `generate_text()` method.

Scoring Stage A's findings directly measures PERCEPTION. Scoring Stage B's
diagnosis (produced from findings that have already been graded) measures
REASONING conditioned on whatever was actually perceived — the two numbers,
together with the existing single-pass evidence_structured F1, let us attribute
error to perception, reasoning, or both.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from _bootstrap import load_cfg

from src import prompts, schema
from src.inference import make_backend, run_inference, _generate_with_retry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_evidence_structured.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=50, help="pilot-first; 0 = full split")
    ap.add_argument("--out-suffix", default="twostage")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--uids-file", default=None,
                    help="CSV with a 'uid' column: restrict to exactly these studies")
    ap.add_argument("--image-variant", default=None,
                    help="Experiment 1: swap in pre-rendered image parts for this "
                         "variant (see scripts/37_build_image_variants.py). Only the "
                         "visual input changes; prompts/model/eval stay fixed.")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out = cfg["paths"]["out_dir"]
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

    # Experiment 1: point the study rows at pre-rendered variant images. Stage B is
    # already image-free, so the visual manipulation propagates to it only through
    # the frozen Stage-A findings -- exactly the intended causal path.
    if args.image_variant:
        from src import imaging
        by_uid = imaging.load_manifest(out, args.image_variant)
        missing = [u for u in df.uid if u not in by_uid]
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(df)} studies have no rendered parts for "
                f"variant '{args.image_variant}' (e.g. {missing[:3]}). Re-run "
                f"37_build_image_variants.py with a matching --limit.")
        df["frontal_paths"] = df.uid.map(lambda u: by_uid[u]["frontal"])
        df["lateral_paths"] = df.uid.map(lambda u: by_uid[u]["lateral"])
        df["image_captions"] = df.uid.map(lambda u: by_uid[u].get("captions", []))
        # Let the backend send every rendered part (arm C sends 7). Defaults stay 1/1
        # for every other config; this only widens the cap to what was rendered.
        mf = max(len(by_uid[u]["frontal"]) for u in df.uid)
        ml = max(len(by_uid[u]["lateral"]) for u in df.uid)
        cfg["inference"]["max_frontal_images"] = max(1, mf)
        cfg["inference"]["max_lateral_images"] = max(1, ml)
        n_parts = sum(len(by_uid[u]["frontal"]) + len(by_uid[u]["lateral"])
                      for u in df.uid)
        print(f"image variant '{args.image_variant}': {n_parts} parts across "
              f"{len(df)} studies (max {mf} frontal-stream + {ml} lateral per call)",
              flush=True)

    # --- Stage A: image -> findings only (reuses run_inference unchanged) ---
    cfg_a = dict(cfg)
    cfg_a["inference"] = dict(cfg["inference"])
    cfg_a["inference"]["prompts"] = ["perception_only"]
    print(f"=== Stage A: perception_only on {len(df)} studies ===")
    ckpt_a = os.path.join(out, f"predictions_{args.split}_{args.out_suffix}_stageA.partial.csv")
    stage_a = run_inference(df, cfg_a, checkpoint=ckpt_a)
    stage_a.to_csv(os.path.join(out, f"predictions_{args.split}_{args.out_suffix}_stageA.csv"),
                   index=False)
    valid_a = stage_a.schema_valid.mean()
    print(f"Stage A done | schema-valid: {100*valid_a:.1f}%")

    # --- Stage B: frozen findings (TEXT ONLY, no image) -> diagnosis ---
    backend = make_backend(cfg)
    if not hasattr(backend, "generate_text"):
        raise SystemExit(f"backend {cfg['inference']['backend']} has no generate_text() "
                         "(text-only Stage B) — two-stage pipeline needs it.")
    row_by_uid = df.set_index("uid")
    stage_a_by_uid = stage_a.set_index("uid")

    print(f"\n=== Stage B: diagnose_from_findings (text-only) ===")

    def work(uid):
        try:
            resp_a = json.loads(stage_a_by_uid.loc[uid, "response"])
        except Exception:
            resp_a = {}
        findings = resp_a.get("findings", {}) if isinstance(resp_a, dict) else {}
        indication = row_by_uid.loc[uid].get("indication", "")
        system, user = prompts.diagnose_from_findings(findings, indication)

        class _TextTask:
            def generate(self, row, cond):
                return backend.generate_text(system, user)
        out_b, err = _generate_with_retry(_TextTask(), None, None)
        errs = schema.validate(out_b) if out_b else ["no output"]
        out_b = out_b or {"primary_diagnosis": "", "confidence": 50, "explanation": "",
                          "error": err}
        return {
            "uid": uid, "prompt_condition": "two_stage_diagnose",
            "stage_a_findings": json.dumps(findings),
            "response": json.dumps(out_b),
            "primary_diagnosis": out_b.get("primary_diagnosis"),
            "confidence": out_b.get("confidence"),
            "schema_valid": len(errs) == 0, "schema_errors": ";".join(errs),
            "api_error": err or "",
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, u): u for u in stage_a_by_uid.index}
        done = 0
        for fut in as_completed(futs):
            records.append(fut.result())
            done += 1
            if done % 25 == 0:
                print(f"  stage B {done}/{len(futs)}", flush=True)

    stage_b = pd.DataFrame(records)
    stage_b.to_csv(os.path.join(out, f"predictions_{args.split}_{args.out_suffix}_stageB.csv"),
                   index=False)
    valid_b = stage_b.schema_valid.mean()
    print(f"Stage B done | schema-valid: {100*valid_b:.1f}%")
    print(f"\nWrote stageA + stageB predictions with suffix '{args.out_suffix}'")


if __name__ == "__main__":
    main()
