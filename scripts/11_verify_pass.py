"""Chain-of-verification pass over targeted predictions (FP attack).

Residual-error audit of structured_targeted: 886 FP vs 351 FN — precision is the
whole remaining game (FP→0 would give F1 0.68). Second pass: show the model its
own positive findings and make it re-examine the image, keeping a finding only if
it can point to specific visual evidence. Only studies with >=1 positive finding
need a second call, and the pass can only REMOVE findings (never add), so recall
is bounded below by the first pass minus rejected true positives.

Writes predictions_test_<suffix>.csv with the verified findings; downstream eval
works unchanged.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from _bootstrap import load_cfg

from src import schema
from src.inference import GeminiBackend, _load_image_bytes, _generate_with_retry

VERIFY_SYSTEM = (
    "You are auditing a preliminary chest radiograph read for over-calls. "
    "You are shown the image(s) and a list of findings a first-pass reader marked "
    "present. For EACH candidate finding, re-examine the image skeptically and "
    "decide: KEEP only if you can point to specific visual evidence at a specific "
    "location; REJECT if the evidence is absent, equivocal, or explainable as "
    "normal anatomy, positioning, or exposure. Over-calling is the known failure "
    "mode: when in doubt, REJECT. Return ONLY valid JSON."
)


def verify_prompt(candidates: list[str]) -> str:
    lst = "\n".join(f"- {c}" for c in candidates)
    return (
        "A first-pass reader marked these findings PRESENT on this chest "
        f"radiograph:\n{lst}\n\n"
        "Re-examine the image for each one. Return JSON: "
        '{"verdicts": {<finding name>: {"decision": "keep"|"reject", '
        '"evidence": "<one sentence: what you see and where, or why rejected>"}}}'
    )


class VerifyBackend(GeminiBackend):
    """Reuses GeminiBackend image/config handling but sends the verify prompt."""

    def generate(self, row: pd.Series, condition: str) -> dict:  # condition = json list
        from google.genai import types
        candidates = json.loads(condition)
        parts = []
        paths = (row.get("frontal_paths") or [])[:1] + (row.get("lateral_paths") or [])[:1]
        for p in paths:
            if os.path.exists(p):
                data, mime = _load_image_bytes(p, self.max_side)
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        parts.append(types.Part.from_text(text=verify_prompt(candidates)))
        gen_kwargs = dict(system_instruction=VERIFY_SYSTEM, temperature=0.0,
                          max_output_tokens=self.max_output_tokens,
                          response_mime_type="application/json")
        try:
            gen_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget)
        except Exception:
            pass
        if self.media_resolution:
            gen_kwargs["media_resolution"] = self.media_resolution
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(**gen_kwargs))
        return schema.extract_json(resp.text or "") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_targeted.yaml")
    ap.add_argument("--preds", default="predictions_test_targeted.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out-suffix", default="verified")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out = cfg["paths"]["out_dir"]
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    for col in ["frontal_paths", "lateral_paths"]:
        studies[col] = studies[col].map(lambda s: eval(s) if isinstance(s, str) else s)
    preds = pd.read_csv(os.path.join(out, args.preds))
    if args.limit:
        preds = preds.head(args.limit)

    backend = VerifyBackend(cfg)

    def positives(resp):
        f = resp.get("findings", {}) or {}
        return [c for c, v in f.items()
                if isinstance(v, dict) and str(v.get("presence", "")).lower() == "present"]

    work_items = []
    for idx, r in preds.iterrows():
        try:
            resp = json.loads(r["response"])
        except Exception:
            continue
        pos = positives(resp)
        if pos:
            work_items.append((idx, r["uid"], resp, pos))
    print(f"{len(preds)} predictions | {len(work_items)} have >=1 positive finding "
          f"-> {len(work_items)} verify calls")

    def do(item):
        idx, uid, resp, pos = item
        row = studies.loc[uid]
        out_obj, err = _generate_with_retry(backend, row, json.dumps(pos))
        verdicts = (out_obj or {}).get("verdicts", {}) or {}
        rejected = []
        for c in pos:
            v = verdicts.get(c)
            if isinstance(v, dict) and str(v.get("decision", "")).lower() == "reject":
                resp["findings"][c]["presence"] = "rejected"
                rejected.append(c)
        kept = [c for c in pos if c not in rejected]
        resp["verify_rejected"] = rejected
        resp["primary_diagnosis"] = (kept[0] if kept
                                     else "No acute cardiopulmonary finding")
        return idx, resp, len(rejected), err

    done = n_rej = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(do, it) for it in work_items]
        for fut in as_completed(futs):
            idx, resp, nr, err = fut.result()
            preds.loc[idx, "response"] = json.dumps(resp)
            preds.loc[idx, "primary_diagnosis"] = resp["primary_diagnosis"]
            n_rej += nr
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(work_items)} verified | findings rejected so far: {n_rej}",
                      flush=True)

    fname = f"predictions_{args.split}_{args.out_suffix}.csv"
    preds.to_csv(os.path.join(out, fname), index=False)
    print(f"Wrote {fname} | total findings rejected: {n_rej}")


if __name__ == "__main__":
    main()
