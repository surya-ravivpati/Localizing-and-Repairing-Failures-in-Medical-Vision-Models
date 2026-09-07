"""VLM inference runner with pluggable backends (protocol §5-6, §14).

Backends:
  - MockBackend: deterministic, offline. Simulates an IMPERFECT model by peeking
    at the report-derived labels with tunable accuracy / hallucination /
    miscalibration, and bakes in modest PROMPT EFFECTS (evidence-first ->
    better groundedness; uncertainty-first -> lower confidence + fewer
    hallucinations + more abstention; cot/ddx -> more verbose, ddx pads the
    differential). This exists ONLY to exercise the evaluation + stats code
    end-to-end without API access. It is NOT a model and must never be reported
    as a result.
  - AnthropicBackend: real vision call to a Claude model. Sends base64 image(s)
    + the prompt, parses JSON. Activated when backend == 'anthropic' and
    ANTHROPIC_API_KEY is set.

All backends return a schema-valid dict (see schema.py).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import prompts, schema
from .labeling import CHEXPERT_CLASSES

PATHOLOGY = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
NORMAL_DX = "No acute cardiopulmonary finding"
FABRICATED = ["12.5 cm cardiac diameter", "compared to prior study",
              "new since previous exam", "2.3 cm nodule", "measured effusion"]


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #
def _seed_for(case_id: str, condition: str, model_id: str) -> int:
    h = hashlib.md5(f"{case_id}|{condition}|{model_id}".encode()).hexdigest()
    return int(h[:8], 16)


def _true_findings(row: pd.Series) -> list[str]:
    return [c for c in PATHOLOGY if row.get(f"lbl_{c}") == 1.0]


# Per-prompt effect multipliers (test harness ground truth for H1-H4).
_PROMPT_FX = {
    #                 acc     halluc  conf_bias  abstain
    "direct":         (0.00,  1.00,   +0.00,     0.00),
    "cot":            (+0.02, 1.05,   +0.03,     0.00),
    "ddx":            (+0.02, 1.20,   +0.02,     0.02),
    "evidence_first": (+0.03, 0.55,   -0.02,     0.05),
    "uncertainty_first": (+0.00, 0.50, -0.15,    0.20),
}


class MockBackend:
    def __init__(self, cfg: dict):
        m = cfg["inference"]["mock"]
        self.base_acc = m["base_accuracy"]
        self.hall = m["hallucination_rate"]
        self.miscal = m["miscalibration"]
        self.model_id = cfg["inference"]["model_id"]

    def generate(self, row: pd.Series, condition: str) -> dict:
        rng = random.Random(_seed_for(str(row["uid"]), condition, self.model_id))
        d_acc, hall_mul, conf_bias, abstain_bonus = _PROMPT_FX[condition]
        acc = min(0.98, self.base_acc + d_acc)
        # Ambiguous cases are harder (drives H4).
        if row.get("difficulty") == "ambiguous":
            acc -= 0.18
        if row.get("difficulty") in ("hard_multi", "rare"):
            acc -= 0.10

        truth = _true_findings(row)
        is_normal = (len(truth) == 0)
        correct = rng.random() < acc

        if is_normal:
            dx = NORMAL_DX if correct else rng.choice(PATHOLOGY)
        else:
            dx = rng.choice(truth) if correct else \
                rng.choice([NORMAL_DX] + [p for p in PATHOLOGY if p not in truth])

        # Confidence: higher when "correct", inflated by miscalibration, shifted by prompt.
        base_conf = 0.80 if correct else 0.62
        conf = base_conf + self.miscal + conf_bias + rng.uniform(-0.08, 0.08)
        conf = int(max(3, min(99, round(conf * 100))))

        # Evidence claims: some true, plus possible hallucinated ones.
        evidence = []
        for f in truth[:3]:
            evidence.append({"finding": f, "location": "", "presence": "present"})
        if not truth:
            evidence.append({"finding": "clear lungs", "location": "bilateral",
                             "presence": "present"})
        hallucinated = rng.random() < (self.hall * hall_mul)
        if hallucinated:
            fab = rng.choice(FABRICATED + [rng.choice(PATHOLOGY)])
            evidence.append({"finding": fab, "location": "", "presence": "present"})

        expl = f"Findings consistent with {dx.lower()}."
        if hallucinated:
            expl += f" Note {evidence[-1]['finding']}."

        out = {
            "primary_diagnosis": dx,
            "confidence": conf,
            "explanation": expl,
            "cited_findings": [e["finding"] for e in evidence if e["presence"] == "present"],
        }

        if condition == "cot":
            out["reasoning"] = ("Lungs reviewed, pleura reviewed, cardiomediastinal "
                                "silhouette assessed, osseous structures inspected. " + expl)
        if condition == "ddx":
            diff = [{"diagnosis": dx, "probability": conf,
                     "rationale": "primary consideration"}]
            # ddx padding: extra plausible-but-maybe-absent entries.
            for _ in range(rng.randint(1, 3)):
                pad = rng.choice(PATHOLOGY)
                diff.append({"diagnosis": pad, "probability": rng.randint(5, 40),
                             "rationale": "considered"})
            out["differential"] = diff
        if condition == "evidence_first":
            out["extracted_evidence"] = evidence
        if condition == "uncertainty_first":
            abstain = (row.get("difficulty") == "ambiguous") and \
                      (rng.random() < (0.4 + abstain_bonus))
            out["uncertainty_statement"] = ("Image quality adequate; some regions "
                                            "partially obscured.")
            out["abstained"] = bool(abstain)
            if abstain:
                out["primary_diagnosis"] = "Indeterminate / insufficient evidence"
                out["confidence"] = int(conf * 0.5)
        return schema.coerce(out)


# --------------------------------------------------------------------------- #
# Anthropic (real VLM) backend
# --------------------------------------------------------------------------- #
class AnthropicBackend:
    def __init__(self, cfg: dict):
        import anthropic  # imported lazily
        self.client = anthropic.Anthropic()
        self.model_id = cfg["inference"]["model_id"]
        self.temperature = cfg["inference"].get("temperature", 0.0)

    @staticmethod
    def _img_block(path: str) -> dict:
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        return {"type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": data}}

    def generate(self, row: pd.Series, condition: str) -> dict:
        system, user = prompts.build(condition, row.get("indication", ""))
        content = []
        for p in (row.get("frontal_paths") or [])[:1] + (row.get("lateral_paths") or [])[:1]:
            if os.path.exists(p):
                content.append(self._img_block(p))
        content.append({"type": "text", "text": user})
        msg = self.client.messages.create(
            model=self.model_id, max_tokens=1024, temperature=self.temperature,
            system=system, messages=[{"role": "user", "content": content}],
        )
        raw = "".join(b.text for b in msg.content if b.type == "text")
        obj = schema.extract_json(raw) or {}
        obj.setdefault("primary_diagnosis", "")
        obj.setdefault("confidence", 50)
        obj.setdefault("explanation", raw[:300])
        obj["raw_response"] = raw
        return schema.coerce(obj)


# --------------------------------------------------------------------------- #
# Gemini (real VLM) backend  — google.genai SDK
# --------------------------------------------------------------------------- #
def _gemini_key() -> str | None:
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def _load_image_bytes(path: str, max_side: int | None = 1024) -> tuple[bytes, str]:
    """Return (png_bytes, mime). If `max_side` is falsy (0/None), send the image
    at FULL resolution (#2) so subtle findings are preserved; otherwise downscale
    the longest side to `max_side` to cut token cost."""
    if not max_side:                       # full-resolution: send original file
        with open(path, "rb") as f:
            return f.read(), "image/png"
    from PIL import Image
    import io
    im = Image.open(path)
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)))
    buf = io.BytesIO()
    im.convert("L").save(buf, format="PNG")
    return buf.getvalue(), "image/png"


class GeminiBackend:
    def __init__(self, cfg: dict):
        from google import genai
        key = _gemini_key()
        if not key:
            raise RuntimeError("No Gemini key (set GEMINI_API_KEY / GOOGLE_API_KEY)")
        self.genai = genai
        self.client = genai.Client(api_key=key)
        self.model_id = cfg["inference"]["model_id"]
        self.temperature = cfg["inference"].get("temperature", 0.0)
        self.max_side = cfg["inference"].get("image_max_side", 1024)
        # #2: media_resolution controls how much image detail Gemini processes
        # (it downsamples by default). HIGH -> more tiles/detail for subtle findings.
        self.media_resolution = cfg["inference"].get("media_resolution")
        self.max_output_tokens = cfg["inference"].get("max_output_tokens", 2048)
        # Disable Gemini 2.5 hidden "thinking" so OUR prompt is the only reasoning
        # scaffold (keeps the Direct-vs-CoT contrast clean) and frees the token
        # budget for the JSON answer. Set thinking_budget: -1 in config to re-enable.
        self.thinking_budget = cfg["inference"].get("thinking_budget", 0)
        # Multi-part input (Experiment 1 arm C). Defaults 1/1 == the historical
        # behaviour, so every existing config is unaffected.
        self.max_frontal = cfg["inference"].get("max_frontal_images", 1)
        self.max_lateral = cfg["inference"].get("max_lateral_images", 1)

    def generate(self, row: pd.Series, condition: str) -> dict:
        from google.genai import types
        system, user = prompts.build(condition, row.get("indication", ""))
        parts = []
        paths = ((row.get("frontal_paths") or [])[:self.max_frontal]
                 + (row.get("lateral_paths") or [])[:self.max_lateral])
        caps = row.get("image_captions") or []
        n_images = 0
        for i, p in enumerate(paths):
            if os.path.exists(p):
                data, mime = _load_image_bytes(p, self.max_side)
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                n_images += 1
                # a crop is uninterpretable without knowing where it came from
                if i < len(caps) and caps[i]:
                    parts.append(types.Part.from_text(text=str(caps[i])))
        parts.append(types.Part.from_text(text=user))
        gen_kwargs = dict(system_instruction=system, temperature=self.temperature,
                          max_output_tokens=self.max_output_tokens,
                          response_mime_type="application/json")
        try:  # thinking_config only exists on 2.5+ models
            gen_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget)
        except Exception:
            pass
        if self.media_resolution:  # e.g. "MEDIA_RESOLUTION_HIGH"
            gen_kwargs["media_resolution"] = self.media_resolution
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(**gen_kwargs),
        )
        raw = resp.text or ""
        obj = schema.extract_json(raw) or {}
        obj.setdefault("primary_diagnosis", "")
        obj.setdefault("confidence", 50)
        obj.setdefault("explanation", raw[:300])
        obj["raw_response"] = raw
        # Provenance for the visual-information ablation: what was actually SENT.
        # Without it there is no way to verify from the CSV that a higher-resolution
        # arm really delivered more visual information (schema allows extra keys).
        obj["_n_image_parts"] = n_images
        usage = getattr(resp, "usage_metadata", None)
        obj["_prompt_tokens"] = getattr(usage, "prompt_token_count", None)
        return schema.coerce(obj)

    def generate_text(self, system: str, user: str) -> dict:
        """TEXT-ONLY call — no image parts sent. Used for the two-stage pipeline's
        Stage B (diagnose from frozen findings): the model must reason from the
        findings text alone, with no way to re-examine the image."""
        from google.genai import types
        gen_kwargs = dict(system_instruction=system, temperature=self.temperature,
                          max_output_tokens=self.max_output_tokens,
                          response_mime_type="application/json")
        try:
            gen_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget)
        except Exception:
            pass
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user)])],
            config=types.GenerateContentConfig(**gen_kwargs),
        )
        raw = resp.text or ""
        obj = schema.extract_json(raw) or {}
        obj.setdefault("primary_diagnosis", "")
        obj.setdefault("confidence", 50)
        obj.setdefault("explanation", raw[:300])
        obj["raw_response"] = raw
        return schema.coerce(obj)


# --------------------------------------------------------------------------- #
# MedGemma (local, Apple-Silicon) backend — mlx-vlm
# --------------------------------------------------------------------------- #
class MedGemmaBackend:
    """Runs a medically fine-tuned Gemma VLM entirely on-device via MLX. No API,
    no cost, no data leaving the machine (so any dataset is compliant). Loads the
    4-bit quantized 4B multimodal model once; NOT thread-safe, so run with
    max_workers: 1 (MLX is already parallel internally)."""

    def __init__(self, cfg: dict):
        from mlx_vlm import load
        self.model_id = cfg["inference"].get("model_id",
                                             "mlx-community/medgemma-4b-it-4bit")
        self.model, self.processor = load(self.model_id)
        self.config = self.model.config
        self.max_output_tokens = cfg["inference"].get("max_output_tokens", 1024)
        self.temperature = cfg["inference"].get("temperature", 0.0)
        # send frontal only by default; the 4B model is most reliable with 1 image
        self.n_images = cfg["inference"].get("medgemma_images", 1)
        # Free-text mode: MedGemma 4B's perception collapses under strict-JSON
        # prompts (defaults to "normal"/hallucinates), but reads well in prose.
        # In this mode we send the free-text scaffold and recover findings with
        # negation-aware normalize_dx. Default ON for medgemma.
        self.free_text = cfg["inference"].get("free_text", True)

    def generate(self, row: pd.Series, condition: str) -> dict:
        from mlx_vlm import generate as mlx_generate, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        if self.free_text:
            system, user = prompts.build_freetext(condition, row.get("indication", ""))
        else:
            system, user = prompts.build(condition, row.get("indication", ""))
        paths = (row.get("frontal_paths") or [])[:1]
        if self.n_images > 1:
            paths += (row.get("lateral_paths") or [])[:1]
        paths = [p for p in paths if p and os.path.exists(p)][: self.n_images]
        n = max(1, len(paths))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        prompt = apply_chat_template(self.processor, self.config, messages,
                                     num_images=n)

        if not self.free_text:
            out = mlx_generate(self.model, self.processor, prompt,
                               image=paths or None, max_tokens=self.max_output_tokens,
                               temperature=self.temperature, verbose=False)
            raw = out.text if hasattr(out, "text") else str(out)
            obj = schema.extract_json(raw) or {}
            obj.setdefault("primary_diagnosis", "")
            obj.setdefault("confidence", 50)
            obj.setdefault("explanation", raw[:300])
            obj["raw_response"] = raw
            return schema.coerce(obj)

        # Free-text: STREAM so we can capture each chosen token's logprob and build
        # a real sequence confidence (generate() only exposes the last token's).
        pieces, chosen_lp = [], []
        for r in stream_generate(self.model, self.processor, prompt,
                                 image=paths or None,
                                 max_tokens=self.max_output_tokens,
                                 temperature=self.temperature):
            if getattr(r, "text", None):
                pieces.append(r.text)
            tok, lp = getattr(r, "token", None), getattr(r, "logprobs", None)
            if tok is not None and lp is not None:
                try:                                   # lp = per-vocab logprobs
                    chosen_lp.append(float(lp[tok]))
                except (TypeError, IndexError):
                    pass
        raw = "".join(pieces)

        # recover CheXpert findings from the prose (negation-aware) as a structured
        # 'findings' object the evaluator reads directly.
        from .normalize import normalize_dx
        pos = {c for c in normalize_dx(raw) if c != "No Finding"}
        # confidence = exp(mean chosen-token logprob) -> 0-100. The model's own
        # generation certainty, not a self-report (which collapses the 4B model).
        if chosen_lp:
            conf = int(round(100 * float(np.exp(np.mean(chosen_lp)))))
        else:
            conf = 50
        conf = max(0, min(100, conf))
        findings = {c: {"presence": "present", "confidence": conf} for c in pos}
        primary = (sorted(pos)[0] if pos else "No acute cardiopulmonary finding")
        return schema.coerce({
            "primary_diagnosis": primary, "confidence": conf,
            "explanation": raw[:400], "findings": findings, "raw_response": raw,
        })

    def generate_text(self, system: str, user: str) -> dict:
        """TEXT-ONLY call (no image) for the two-stage Stage B: diagnose from the
        frozen Stage-A findings text alone. Mirrors the free-text streaming path
        (prose-parsed diagnosis + logprob confidence) but with num_images=0, so the
        model cannot re-examine the radiograph — the reasoning is isolated."""
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from .normalize import normalize_dx
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        prompt = apply_chat_template(self.processor, self.config, messages,
                                     num_images=0)
        pieces, chosen_lp = [], []
        for r in stream_generate(self.model, self.processor, prompt, image=None,
                                 max_tokens=self.max_output_tokens,
                                 temperature=self.temperature):
            if getattr(r, "text", None):
                pieces.append(r.text)
            tok, lp = getattr(r, "token", None), getattr(r, "logprobs", None)
            if tok is not None and lp is not None:
                try:
                    chosen_lp.append(float(lp[tok]))
                except (TypeError, IndexError):
                    pass
        raw = "".join(pieces)
        conf = int(round(100 * float(np.exp(np.mean(chosen_lp))))) if chosen_lp else 50
        conf = max(0, min(100, conf))
        # Stage B is text-only REASONING; unlike perception, MedGemma emits clean
        # JSON here, so honor its COMMITTED primary_diagnosis. Prose-mining the whole
        # response would scrape negated findings out of the explanation ("did not
        # show ... cardiomegaly, edema") and manufacture false positives. Fall back
        # to prose only when no parseable diagnosis was returned.
        obj = schema.extract_json(raw) or {}
        primary = str(obj.get("primary_diagnosis", "") or "").strip()
        explanation = str(obj.get("explanation", "") or "").strip() or raw[:400]
        if not primary:
            pos = {c for c in normalize_dx(raw) if c != "No Finding"}
            primary = (sorted(pos)[0] if pos else "No acute cardiopulmonary finding")
        return schema.coerce({
            "primary_diagnosis": primary, "confidence": conf,
            "explanation": explanation, "raw_response": raw,
        })


def make_backend(cfg: dict):
    b = cfg["inference"]["backend"]
    if b == "mock":
        return MockBackend(cfg)
    if b == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set for anthropic backend")
        return AnthropicBackend(cfg)
    if b == "gemini":
        return GeminiBackend(cfg)
    if b == "medgemma":
        return MedGemmaBackend(cfg)
    raise ValueError(f"unknown backend {b}")


def _generate_with_retry(backend, row, cond, retries: int = 5):
    """Retry with exponential backoff; longer waits on rate-limit (429) errors."""
    import time
    last = None
    for attempt in range(retries + 1):
        try:
            return backend.generate(row, cond), None
        except Exception as e:
            last = str(e)
            if attempt < retries:
                is_rate = any(s in last.lower() for s in
                              ("429", "rate", "quota", "resource_exhausted"))
                base = 8.0 if is_rate else 2.0
                time.sleep(min(60.0, base * (2 ** attempt)))
    return None, last


def _record(row, cond, cfg, out, err):
    errs = schema.validate(out) if out else ["no output"]
    out = out or {"primary_diagnosis": "", "confidence": 50, "explanation": "",
                  "error": err}
    return {
        "uid": row["uid"], "case_id": str(row["uid"]), "prompt_condition": cond,
        "model_id": cfg["inference"]["model_id"],
        "prompt_version": prompts.PROMPT_VERSION, "difficulty": row.get("difficulty"),
        "response": json.dumps(out), "primary_diagnosis": out.get("primary_diagnosis"),
        "confidence": out.get("confidence"), "schema_valid": len(errs) == 0,
        "schema_errors": ";".join(errs), "api_error": err or "",
        "n_image_parts": out.get("_n_image_parts"),
        "prompt_tokens": out.get("_prompt_tokens"),
    }


def run_inference(df: pd.DataFrame, cfg: dict, verbose: bool = True,
                  checkpoint: str | None = None) -> pd.DataFrame:
    """Concurrent, checkpointed inference.

    - max_workers from cfg['inference']['max_workers'] (default 8).
    - If `checkpoint` exists, already-done (uid, prompt) pairs are skipped so a
      long run can resume after interruption. Partial results flush periodically.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    backend = make_backend(cfg)
    conditions = cfg["inference"]["prompts"]
    workers = cfg["inference"].get("max_workers", 8)

    done_keys, records = set(), []
    if checkpoint and os.path.exists(checkpoint):
        prev = pd.read_csv(checkpoint)
        records = prev.to_dict("records")
        done_keys = {(r["uid"], r["prompt_condition"]) for r in records}
        if verbose:
            print(f"  resuming: {len(done_keys)} calls already done", flush=True)

    tasks = [(row, cond) for _, row in df.iterrows() for cond in conditions
             if (row["uid"], cond) not in done_keys]
    total = len(tasks) + len(done_keys)
    lock = threading.Lock()
    done = len(done_keys)

    def work(task):
        row, cond = task
        out, err = _generate_with_retry(backend, row, cond)
        return _record(row, cond, cfg, out, err)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                records.append(rec)
                done += 1
                if verbose and (done % 25 == 0 or done == total):
                    n_err = sum(1 for r in records if r["api_error"])
                    print(f"  {done}/{total} calls | errors: {n_err}", flush=True)
                if checkpoint and done % 50 == 0:
                    pd.DataFrame(records).to_csv(checkpoint, index=False)
    out_df = pd.DataFrame(records)
    if checkpoint:
        out_df.to_csv(checkpoint, index=False)
    return out_df
