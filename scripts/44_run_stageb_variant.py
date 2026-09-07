"""Experiment 5 — re-run Stage B alone against an EXISTING Stage-A run.

The 2x2 crosses visual information (which Stage-A run supplies the findings)
with reasoning effort (which Stage-B prompt consumes them). Because Stage B is
text-only and Stage A is already on disk, the visual factor costs nothing to vary
and the perception stage is byte-identical across reasoning levels — the cleanest
possible separation of the two factors.

  python3 scripts/44_run_stageb_variant.py --stage-a exp3_blur12_f --reasoning cot
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from _bootstrap import load_cfg

from src import prompts, schema
from src.inference import make_backend, _generate_with_retry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_exp1_gemini.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--stage-a", required=True, help="suffix of an existing stageA file")
    ap.add_argument("--reasoning", choices=["direct", "cot"], default="cot")
    ap.add_argument("--out-suffix", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out = cfg["paths"]["out_dir"]
    sa_path = os.path.join(out, f"predictions_{args.split}_{args.stage_a}_stageA.csv")
    if not os.path.exists(sa_path):
        raise SystemExit(f"missing {sa_path}")
    stage_a = pd.read_csv(sa_path).set_index("uid")
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    uids = [u for u in stage_a.index if u in studies.index]
    suffix = args.out_suffix or f"exp5_{args.stage_a}_{args.reasoning}"

    backend = make_backend(cfg)
    builder = (prompts.diagnose_from_findings_cot if args.reasoning == "cot"
               else prompts.diagnose_from_findings)
    print(f"Stage B ({args.reasoning}) over {len(uids)} frozen Stage-A reads "
          f"from '{args.stage_a}'", flush=True)

    def work(uid):
        try:
            resp_a = json.loads(stage_a.loc[uid, "response"])
        except Exception:
            resp_a = {}
        findings = resp_a.get("findings", {}) if isinstance(resp_a, dict) else {}
        system, user = builder(findings, studies.loc[uid].get("indication", ""))

        class _T:
            def generate(self, row, cond):
                return backend.generate_text(system, user)
        o, err = _generate_with_retry(_T(), None, None)
        errs = schema.validate(o) if o else ["no output"]
        o = o or {"primary_diagnosis": "", "confidence": 50, "explanation": "", "error": err}
        return {"uid": uid, "prompt_condition": f"stageb_{args.reasoning}",
                "stage_a_findings": json.dumps(findings), "response": json.dumps(o),
                "primary_diagnosis": o.get("primary_diagnosis"),
                "confidence": o.get("confidence"), "schema_valid": len(errs) == 0,
                "schema_errors": ";".join(errs), "api_error": err or ""}

    recs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, u): u for u in uids}
        for i, f in enumerate(as_completed(futs), 1):
            recs.append(f.result())
            if i % 50 == 0:
                print(f"  {i}/{len(uids)}", flush=True)
    df = pd.DataFrame(recs)
    dest = os.path.join(out, f"predictions_{args.split}_{suffix}_stageB.csv")
    df.to_csv(dest, index=False)
    print(f"schema-valid: {100*df.schema_valid.mean():.1f}% | wrote {dest}")


if __name__ == "__main__":
    main()
