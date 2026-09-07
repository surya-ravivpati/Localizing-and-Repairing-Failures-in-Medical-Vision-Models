"""CheXbert reference labeler (Smit et al. 2020) — the field-standard model for
labeling chest-X-ray reports into the 14 CheXpert findings.

Drop-in replacement for the rule-based labeling.label_report. We reuse the
CheXbert model + tokenizer from the `f1chexbert` package but re-implement the
tokenization with the modern transformers API (the package's `encode_plus` path
is broken on transformers>=5) and batch it for speed.

CheXbert per-class output states -> our convention:
    0 blank/not-mentioned -> NaN
    1 positive            -> 1.0
    2 negative            -> 0.0
    3 uncertain           -> -1.0
"""
from __future__ import annotations

import numpy as np
import torch

_M = None


def _model():
    global _M
    if _M is None:
        from f1chexbert import F1CheXbert
        _M = F1CheXbert(device="cpu")
    return _M


_STATE_MAP = {0: np.nan, 1: 1.0, 2: 0.0, 3: -1.0}


def label_reports_chexbert(texts: list[str], batch_size: int = 32,
                           progress: bool = False) -> list[dict]:
    """Label many reports. Returns a list of {class_name: state} dicts, keyed by
    CheXbert's target_names (14 classes, 'No Finding' last)."""
    from f1chexbert.f1chexbert import generate_attention_masks
    m = _model()
    tok = m.tokenizer
    names = m.target_names
    results: list[dict] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts[i:i + batch_size]
        seqs = []
        for t in chunk:
            t = (str(t) if t is not None else "").strip()
            # modern transformers API: tokenizer(...) adds [CLS]/[SEP] + truncates
            ids = tok(t, truncation=True, max_length=512,
                      add_special_tokens=True)["input_ids"]
            if not ids:
                ids = [tok.cls_token_id, tok.sep_token_id]
            seqs.append(ids)
        maxlen = max(len(s) for s in seqs)
        padded = [s + [tok.pad_token_id] * (maxlen - len(s)) for s in seqs]
        batch = torch.LongTensor(padded)
        src_len = [len(s) for s in seqs]
        attn = generate_attention_masks(batch, src_len, m.device)
        with torch.no_grad():
            out = m.model(batch.to(m.device), attn)   # list[14] of (B, states)
        preds = [o.argmax(dim=1).tolist() for o in out]
        for b in range(len(chunk)):
            results.append({names[ci]: _STATE_MAP.get(preds[ci][b], np.nan)
                            for ci in range(len(names))})
        if progress:
            print(f"  chexbert {min(i + batch_size, n)}/{n}", flush=True)
    return results


def label_report_chexbert(text: str) -> dict:
    return label_reports_chexbert([text])[0]


if __name__ == "__main__":
    demo = ["Basilar atelectasis. No pleural effusion.",
            "No acute cardiopulmonary abnormality.",
            "Large right pleural effusion. Findings may represent pneumonia."]
    for t, lab in zip(demo, label_reports_chexbert(demo)):
        pos = {k: v for k, v in lab.items() if not (isinstance(v, float) and np.isnan(v))}
        print(f"{t[:50]!r}\n  {pos}\n")
