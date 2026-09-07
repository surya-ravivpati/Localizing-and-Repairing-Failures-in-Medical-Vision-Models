"""LLM-as-a-Judge harness with bias controls (protocol §8).

Rubric-based absolute scoring (1-5) on 7 dimensions, with faithfulness kept
SEPARATE from persuasiveness/coherence so the H3 dissociation is measurable.
Backends:
  - MockJudge: deterministic; derives scores from the automated groundedness
    signal + a verbosity term, so 'persuasiveness' rises with length while
    'faithfulness' tracks support-rate (reproduces H3 in the test harness).
  - AnthropicJudge: real judge call (blind, JSON rubric). Use a DIFFERENT model
    family than the subject model to reduce self-preference bias.

Pairwise mode runs both orders (position-bias control) and reports agreement.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

DIMENSIONS = ["evidence_identification", "logical_consistency", "ddx_quality",
              "appropriate_uncertainty", "faithfulness", "clinical_usefulness",
              "diagnostic_coherence"]

JUDGE_SYSTEM = (
    "You are a careful evaluator scoring a chest X-ray interpretation for a research "
    "study. You are given (A) a reference radiology report, (B) reference structured "
    "findings, and (C) a candidate interpretation. Score ONLY the candidate. Do NOT "
    "reward length or confident tone. Faithfulness = whether claims are supported by "
    "the reference, independent of how persuasive the writing is."
)

JUDGE_RUBRIC = """Score each dimension 1-5:
- evidence_identification: 5=all key reference findings addressed; 1=misses all.
- logical_consistency:     5=no internal contradictions; 1=self-contradictory.
- ddx_quality:             5=appropriate, correctly ranked; 1=irrelevant/absent.
- appropriate_uncertainty: 5=uncertainty matches evidence & report hedging; 1=miscalibrated.
- faithfulness:            5=every claim supported/uncontradicted; 1=multiple contradictions.
- clinical_usefulness:     5=actionable & correct; 1=misleading.
- diagnostic_coherence:    5=evidence->dx follows; 1=non-sequitur.
Also flag contradicted_claims[] and fabricated_specifics[].
Return JSON: {"scores":{...seven keys...},"contradicted_claims":[],"fabricated_specifics":[],"justification":"<=60 words"}"""


def build_judge_user(report_text: str, ref_findings: list[str], candidate: dict) -> str:
    return (f"(A) REFERENCE REPORT:\n{report_text}\n\n"
            f"(B) REFERENCE FINDINGS: {', '.join(ref_findings) or 'None (normal)'}\n\n"
            f"(C) CANDIDATE INTERPRETATION:\n{json.dumps(candidate)}\n\n{JUDGE_RUBRIC}")


class MockJudge:
    def __init__(self, judge_id: str):
        self.judge_id = judge_id

    def score(self, report_text, ref_findings, candidate, groundedness_row) -> dict:
        rng = np.random.RandomState(
            int(hashlib.md5((self.judge_id + str(candidate)).encode()).hexdigest()[:8], 16))
        # groundedness_row may be a dict-of-dicts when the groundedness frame has a
        # MultiIndex slice; coerce defensively so the mock never raises (it used to
        # crash with "unsupported operand type(s) for *: 'int' and 'dict'").
        sr = (groundedness_row or {}).get("support_rate", 0.5)
        if isinstance(sr, dict):
            sr = next((v for v in sr.values() if isinstance(v, (int, float))), 0.5)
        try:
            sr = float(sr)
        except (TypeError, ValueError):
            sr = 0.5
        sr = 0.5 if np.isnan(sr) else sr
        length = len(json.dumps(candidate))
        verbosity = min(1.0, length / 600.0)          # persuasiveness proxy
        faith = 1 + 4 * sr                              # tracks support-rate
        persuasive = 1 + 4 * (0.4 * sr + 0.6 * verbosity)
        noise = lambda: rng.uniform(-0.4, 0.4)
        scores = {
            "faithfulness": faith + noise(),
            "evidence_identification": faith + noise(),
            "logical_consistency": persuasive + noise(),
            "diagnostic_coherence": persuasive + noise(),
            "clinical_usefulness": 0.5 * (faith + persuasive) + noise(),
            "ddx_quality": (persuasive if candidate.get("differential") else 2.5) + noise(),
            "appropriate_uncertainty": (4.0 if candidate.get("abstained") is not None
                                        else 2.8) + noise(),
        }
        scores = {k: float(np.clip(round(v, 1), 1, 5)) for k, v in scores.items()}
        return {"judge_id": self.judge_id, "scores": scores,
                "contradicted_claims": [], "fabricated_specifics": []}


class AnthropicJudge:
    def __init__(self, judge_id: str, model_id: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.judge_id = judge_id
        self.model_id = model_id

    def score(self, report_text, ref_findings, candidate, groundedness_row=None) -> dict:
        user = build_judge_user(report_text, ref_findings, candidate)
        msg = self.client.messages.create(
            model=self.model_id, max_tokens=600, temperature=0.0,
            system=JUDGE_SYSTEM, messages=[{"role": "user", "content": user}])
        raw = "".join(b.text for b in msg.content if b.type == "text")
        try:
            obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        except Exception:
            obj = {"scores": {d: np.nan for d in DIMENSIONS}}
        obj["judge_id"] = self.judge_id
        return obj


class GeminiJudge:
    def __init__(self, judge_id: str, model_id: str):
        from google import genai
        from src.inference import _gemini_key
        key = _gemini_key()
        if not key:
            raise RuntimeError("No Gemini key for judge")
        self.client = genai.Client(api_key=key)
        self.judge_id = judge_id
        self.model_id = model_id

    def score(self, report_text, ref_findings, candidate, groundedness_row=None) -> dict:
        """Score one candidate.

        Fixed 2026-07-19 (judge produced usable scores for only 1.3% of outputs):
          * thinking_budget=0 — Gemini 2.5's hidden CoT was consuming the whole
            output budget, so `resp.text` came back empty/truncated. This is the
            same fix already applied in inference.GeminiBackend.
          * max_output_tokens 700 -> 1500 (7 scores + 2 lists + justification).
          * retry with backoff instead of one-shot.
          * failures are RECORDED (judge_error) rather than silently becoming NaN.
        """
        import time
        from google.genai import types
        user = build_judge_user(report_text, ref_findings, candidate)
        gen_kwargs = dict(system_instruction=JUDGE_SYSTEM, temperature=0.0,
                          max_output_tokens=1500,
                          response_mime_type="application/json")
        try:
            gen_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass

        last_err = None
        for attempt in range(4):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id, contents=user,
                    config=types.GenerateContentConfig(**gen_kwargs))
                raw = resp.text or ""
                if not raw.strip():
                    fr = (resp.candidates[0].finish_reason
                          if getattr(resp, "candidates", None) else "?")
                    raise ValueError(f"empty response (finish_reason={fr})")
                obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                if not isinstance(obj.get("scores"), dict) or not obj["scores"]:
                    raise ValueError("no scores object in response")
                obj["judge_id"] = self.judge_id
                obj["judge_error"] = ""
                return obj
            except Exception as e:
                last_err = str(e)[:200]
                if attempt < 3:
                    is_rate = any(t in last_err.lower() for t in
                                  ("429", "rate", "quota", "resource_exhausted"))
                    time.sleep(min(45.0, (8.0 if is_rate else 2.0) * (2 ** attempt)))
        return {"scores": {d: np.nan for d in DIMENSIONS},
                "judge_id": self.judge_id, "judge_error": last_err or "unknown"}


def make_judges(cfg: dict):
    backend = cfg["judge"]["backend"]
    if backend == "mock":
        return [MockJudge(j) for j in cfg["judge"]["models"]]
    if backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set for anthropic judge")
        return [AnthropicJudge(j, j) for j in cfg["judge"]["models"]]
    if backend == "gemini":
        return [GeminiJudge(j, j) for j in cfg["judge"]["models"]]
    raise ValueError(f"unknown judge backend {backend}")


def run_judging(pred_df, study_df, ground_df, cfg, verbose=True,
                checkpoint=None) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    from .labeling import CHEXPERT_CLASSES

    judges = make_judges(cfg)
    workers = cfg["judge"].get("max_workers", 8)
    truth = study_df.set_index("uid")
    gidx = ground_df.set_index(["uid", "prompt_condition"])

    done_keys, rows = set(), []
    if checkpoint and os.path.exists(checkpoint):
        prev = pd.read_csv(checkpoint)
        rows = prev.to_dict("records")
        done_keys = {(r["uid"], r["prompt_condition"], r["judge_id"]) for r in rows}
        if verbose:
            print(f"  judge resuming: {len(done_keys)} done", flush=True)

    tasks = []
    for _, r in pred_df.iterrows():
        for j in judges:
            if (r["uid"], r["prompt_condition"], j.judge_id) in done_keys:
                continue
            tasks.append((r, j))
    total = len(tasks) + len(done_keys)
    lock = threading.Lock()
    done = len(done_keys)

    def work(task):
        r, j = task
        resp = json.loads(r["response"])
        ref_row = truth.loc[r["uid"]]
        ref_findings = [c for c in CHEXPERT_CLASSES
                        if c != "No Finding" and ref_row.get(f"lbl_{c}") == 1.0]
        try:
            g_row = gidx.loc[(r["uid"], r["prompt_condition"])].to_dict()
        except KeyError:
            g_row = {}
        res = j.score(ref_row.get("report_text", ""), ref_findings, resp, g_row)
        rec = {"uid": r["uid"], "prompt_condition": r["prompt_condition"],
               "difficulty": r["difficulty"], "judge_id": res["judge_id"],
               # persist failures so a silent all-NaN run can never look like a
               # completed one again (the 2026-07-19 judge failure)
               "judge_error": res.get("judge_error", "")}
        scores = res.get("scores", {}) or {}
        rec.update({f"score_{d}": scores.get(d, np.nan) for d in DIMENSIONS})
        rec["n_contradicted"] = len(res.get("contradicted_claims", []) or [])
        rec["n_fabricated"] = len(res.get("fabricated_specifics", []) or [])
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futures):
            with lock:
                rows.append(fut.result())
                done += 1
                if verbose and (done % 25 == 0 or done == total):
                    print(f"  judge {done}/{total}", flush=True)
                if checkpoint and done % 50 == 0:
                    pd.DataFrame(rows).to_csv(checkpoint, index=False)
    return pd.DataFrame(rows)
