"""VisualCheXbert reference labeler (Jain et al. 2021) — the third, image-adjusted
labeler for the reference-confidence layer.

VisualCheXbert corrects the report-vs-image mismatch that plain CheXbert inherits:
it is trained so its predictions correlate with what a radiologist would label from
the IMAGE, not just what the report text says. Architecture (verified against the
official source, stanfordmlgroup/VisualCheXbert, 2026-08-11 — it is NOT the same
model class as CheXbert, despite the similar name):

  1. `bert_labeler`: bert-base-uncased backbone + 14 independent binary linear heads
     (one scalar logit per CheXpert class, sigmoid -> probability). This differs from
     CheXbert's bert_labeler, which has 4-way heads for uncertainty-aware labeling.
  2. A per-class logistic-regression mapping (`logreg_models.pickle`): each of the 14
     LogisticRegression models takes ALL 14 raw BERT probabilities as features and
     outputs the final binary VisualCheXbert label for its class. This second stage
     is the actual "visual" correction, learned to match radiologist image reads —
     skipping it would silently produce CheXbert-like report labels instead.

INSTALL: download & unzip the official checkpoint (Google Drive link in
https://github.com/stanfordmlgroup/VisualCheXbert) to
~/Library/Caches/visualchexbert/checkpoint/{visualCheXbert.pth,logreg_models.pickle}
(or set VISUALCHEXBERT_DIR to the folder containing both files). If unavailable,
`available()` returns False and the reference-confidence layer degrades cleanly to
the two existing labelers (rule-based + CheXbert).
"""
from __future__ import annotations

import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 14 CheXpert observations, in the order VisualCheXbert's heads and logreg models
# expect (matches CHEXPERT_CLASSES elsewhere in this project).
TARGET_NAMES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]

_DIR_CANDIDATES = [
    os.environ.get("VISUALCHEXBERT_DIR", ""),
    os.path.expanduser("~/Library/Caches/visualchexbert/checkpoint"),
    os.path.expanduser("~/.cache/visualchexbert/checkpoint"),
]

_M = None      # (model, tokenizer, logreg_models) once loaded


def _ckpt_dir() -> str | None:
    for d in _DIR_CANDIDATES:
        if d and os.path.exists(os.path.join(d, "visualCheXbert.pth")) \
              and os.path.exists(os.path.join(d, "logreg_models.pickle")):
            return d
    return None


def available() -> bool:
    """True iff both checkpoint files are present so labeling can actually run."""
    return _ckpt_dir() is not None


class _BertLabeler(nn.Module):
    """Mirrors stanfordmlgroup/VisualCheXbert's src/models/bert_labeler.py exactly,
    so the released state_dict loads without remapping."""

    def __init__(self):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        hidden = self.bert.pooler.dense.in_features
        self.linear_heads = nn.ModuleList(
            [nn.Linear(hidden, 1, bias=True) for _ in range(14)])

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids, attention_mask=attention_mask)[0]
        cls = self.dropout(h[:, 0, :])
        return [head(cls).squeeze(-1) for head in self.linear_heads]


def _model():
    global _M
    if _M is None:
        d = _ckpt_dir()
        if d is None:
            raise FileNotFoundError(
                "VisualCheXbert checkpoint not found. Download it from "
                "https://github.com/stanfordmlgroup/VisualCheXbert and place "
                "visualCheXbert.pth + logreg_models.pickle in "
                "~/Library/Caches/visualchexbert/checkpoint/ (or set VISUALCHEXBERT_DIR).")
        from transformers import BertTokenizer
        model = _BertLabeler()
        state = torch.load(os.path.join(d, "visualCheXbert.pth"), map_location="cpu")
        state = state.get("model_state_dict", state)
        # the released checkpoint was trained with nn.DataParallel -> 'module.' prefix
        cleaned = OrderedDict((k[7:] if k.startswith("module.") else k, v)
                              for k, v in state.items())
        model.load_state_dict(cleaned)
        model.eval()
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
        with open(os.path.join(d, "logreg_models.pickle"), "rb") as f:
            logreg = pickle.load(f)
        _M = (model, tok, logreg)
    return _M


def label_reports_visualchexbert(texts: list[str], batch_size: int = 16,
                                 progress: bool = False) -> list[dict]:
    """Label reports with VisualCheXbert. Returns [{class: 1.0/0.0}], keyed by
    TARGET_NAMES. Binary labeler (no uncertain state, unlike CheXbert)."""
    model, tok, logreg = _model()
    n = len(texts)
    all_probs = []
    for i in range(0, n, batch_size):
        chunk = [(str(t) if t is not None else "").strip() for t in texts[i:i + batch_size]]
        enc = [tok(t, truncation=True, max_length=512, add_special_tokens=True)["input_ids"]
               if t else [tok.cls_token_id, tok.sep_token_id] for t in chunk]
        maxlen = max(len(e) for e in enc)
        padded = [e + [tok.pad_token_id] * (maxlen - len(e)) for e in enc]
        input_ids = torch.LongTensor(padded)
        attn = (input_ids != tok.pad_token_id).long()
        attn[:, 0] = 1   # [CLS] is never padding even if pad_token_id collides
        with torch.no_grad():
            out = model(input_ids, attn)               # list[14] of (B,)
        probs = torch.stack([torch.sigmoid(o) for o in out], dim=1).numpy()  # (B,14)
        all_probs.append(probs)
        if progress:
            print(f"  visualchexbert (bert) {min(i + batch_size, n)}/{n}", flush=True)
    probs = np.concatenate(all_probs, axis=0)
    df_probs = pd.DataFrame(probs, columns=TARGET_NAMES)

    # stage 2: per-class logistic regression maps the 14-dim prob vector -> label
    results = {name: logreg[name].predict(df_probs) for name in TARGET_NAMES}
    out = []
    for i in range(n):
        out.append({name: float(results[name][i]) for name in TARGET_NAMES})
    return out


def label_report_visualchexbert(text: str) -> dict:
    return label_reports_visualchexbert([text])[0]


if __name__ == "__main__":
    demo = ["Basilar atelectasis. No pleural effusion.",
            "No acute cardiopulmonary abnormality.",
            "Large right pleural effusion. Findings may represent pneumonia."]
    for t, lab in zip(demo, label_reports_visualchexbert(demo, progress=True)):
        pos = {k: v for k, v in lab.items() if v == 1.0}
        print(f"{t[:50]!r}\n  {pos}\n")
