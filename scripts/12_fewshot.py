"""Few-shot in-context learning with labeled exemplar images (bias attack #2).

The model over/under-calls because it lacks a calibrated internal threshold for
what each finding "looks like" on THIS collection. Few-shot gives it concrete
reference points: a handful of dev-split X-rays (NO test leakage) each paired with
its verified label, then the test image with the targeted classification task.

Exemplars are chosen (script 11-select step) as clear single-finding cases where
bronze AND CheXbert agree exactly. Writes predictions_test_<suffix>.csv.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from _bootstrap import load_cfg

from src import schema
from src.inference import GeminiBackend, _load_image_bytes, _generate_with_retry
from src.prompts import build, STRUCTURED_FINDINGS, _OVERCALL_GUIDANCE, SYSTEM_PREAMBLE

EXEMPLAR_LABEL = {
    "__NORMAL__": "a NORMAL chest radiograph with no acute cardiopulmonary finding",
    "Cardiomegaly": "Cardiomegaly (enlarged cardiac silhouette, cardiothoracic ratio > 0.5)",
    "Lung Opacity": "a Lung Opacity",
    "Atelectasis": "Atelectasis (linear/band-like opacity with volume loss)",
    "Pleural Effusion": "a Pleural Effusion (blunted costophrenic angle / layering fluid)",
    "Support Devices": "Support Devices (visible tube/line/catheter/pacemaker)",
    "Consolidation": "Consolidation (dense airspace opacity with air bronchograms)",
}


class FewShotBackend(GeminiBackend):
    def __init__(self, cfg, exemplars):
        super().__init__(cfg)
        # pre-load exemplar image bytes once (reused for every test image)
        self.ex = []
        for key, (uid, path) in exemplars.items():
            if os.path.exists(path):
                data, mime = _load_image_bytes(path, self.max_side)
                self.ex.append((EXEMPLAR_LABEL.get(key, key), data, mime))

    def generate(self, row, condition):
        from google.genai import types
        # Framing fix: the exemplars are a VISUAL GLOSSARY (what each finding looks
        # like), NOT a severity bar. The v1 design collapsed recall 0.60->0.13
        # because clear single-finding exemplars taught the model to only call
        # obvious cases. Emphasize real findings are subtler than these references.
        parts = [types.Part.from_text(text=(
            "Below is a visual glossary: one CLEAR textbook example of each finding, "
            "to remind you what it looks like. These are deliberately obvious cases. "
            "On real studies the same findings are usually MUCH more subtle than "
            "these examples — do not require the test image to look this clear."))]
        for label, data, mime in self.ex:
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
            parts.append(types.Part.from_text(text=f"^ Reference example of {label}."))

        checklist = ", ".join(STRUCTURED_FINDINGS)
        task = (
            "\nNow assess the following NEW chest radiograph. Systematically assess "
            f"for EACH of: {checklist}.\n"
            "IMPORTANT: mark a finding 'present' whenever you see ANY visible "
            "evidence consistent with it, even if far milder or less obvious than "
            "the reference examples above. The examples show severe cases; most real "
            "positives are subtle. Use 'uncertain' only when genuinely undecidable, "
            "'absent' only when the region truly looks normal.\n"
            f"{_OVERCALL_GUIDANCE}"
            'Return JSON with: "findings" (object mapping each finding name to '
            '{"presence":"present"|"absent"|"uncertain","confidence":int}), '
            '"primary_diagnosis" (string), "confidence" (int), "explanation" (string).')
        paths = (row.get("frontal_paths") or [])[:1] + (row.get("lateral_paths") or [])[:1]
        for p in paths:
            if os.path.exists(p):
                data, mime = _load_image_bytes(p, self.max_side)
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        parts.append(types.Part.from_text(text=task))

        gen_kwargs = dict(system_instruction=SYSTEM_PREAMBLE, temperature=self.temperature,
                          max_output_tokens=self.max_output_tokens,
                          response_mime_type="application/json")
        try:
            gen_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        except Exception:
            pass
        if self.media_resolution:
            gen_kwargs["media_resolution"] = self.media_resolution
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(**gen_kwargs))
        obj = schema.extract_json(resp.text or "") or {}
        obj.setdefault("primary_diagnosis", "")
        obj.setdefault("confidence", 50)
        obj.setdefault("explanation", "")
        return schema.coerce(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_targeted.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out-suffix", default="fewshot")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out = cfg["paths"]["out_dir"]
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    manifest = pd.read_csv(os.path.join(out, f"{args.split}_manifest.csv"))
    df = studies[studies.uid.isin(manifest.uid)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    for col in ["frontal_paths", "lateral_paths"]:
        df[col] = df[col].map(lambda s: eval(s) if isinstance(s, str) else s)

    exemplars = {k: tuple(v) for k, v in
                 json.load(open(os.path.join(out, "fewshot_exemplars.json"))).items()}
    backend = FewShotBackend(cfg, exemplars)
    print(f"few-shot: {len(df)} studies, {len(backend.ex)} exemplar images/call | "
          f"workers={args.workers}")

    rows = [None] * len(df)

    def work(i):
        row = df.iloc[i]
        out_obj, err = _generate_with_retry(backend, row, "fewshot")
        out_obj = out_obj or {"primary_diagnosis": "", "confidence": 50,
                              "explanation": "", "error": err}
        return i, {
            "uid": row["uid"], "case_id": str(row["uid"]), "prompt_condition": "fewshot",
            "model_id": cfg["inference"]["model_id"], "difficulty": row.get("difficulty"),
            "response": json.dumps(out_obj),
            "primary_diagnosis": out_obj.get("primary_diagnosis"),
            "confidence": out_obj.get("confidence"),
            "schema_valid": len(schema.validate(out_obj)) == 0,
            "schema_errors": "", "api_error": err or "",
        }

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(df))]
        for fut in as_completed(futs):
            i, rec = fut.result()
            rows[i] = rec
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(df)}", flush=True)

    preds = pd.DataFrame(rows)
    fname = f"predictions_{args.split}_{args.out_suffix}.csv"
    preds.to_csv(os.path.join(out, fname), index=False)
    n_err = (preds["api_error"] != "").sum()
    print(f"Wrote {fname} | schema-valid {preds.schema_valid.mean()*100:.1f}% | errors {n_err}")


if __name__ == "__main__":
    main()
