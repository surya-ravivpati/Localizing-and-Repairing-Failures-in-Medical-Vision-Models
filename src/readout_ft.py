"""Shared pieces of the per-finding readout fine-tune (scripts 56 train / 57 eval).

Everything that must be IDENTICAL between training and evaluation lives here, so
parity holds by construction rather than by two copies being kept in sync (the old
40/41 pair duplicated the instruction and drifted once already).

WHY THIS EXISTS -- the two causes of every earlier fine-tune failure:

1. NO CAUSAL MASK IN TRAINING (the bug). mlx_vlm's stock `vision_language_loss_fn`
   passes the 2-D padding mask to Gemma-3's `get_input_embeddings`, which turns it
   into mask (x) mask = an ALL-ONES 4-D mask, and the language model then uses that
   instead of its causal mask. Every training position could see the token it was
   asked to predict. Verified directly: editing one late token moved logits at
   EARLIER positions by 37 on the training path and by exactly 0 on the causal
   path. That is why loss fell to ~1e-5 on data that cannot have been memorised,
   and why generation (which IS causal) came out as `model\\nmodel\\n...`.
   `causal_logits` below is the fix: embeddings from `get_input_embeddings`, then
   the language model with mask=None, so it builds its own causal / sliding-window
   masks exactly as it does at generation time.

2. THE OBJECTIVE DID NOT MATCH THE DEFECT. Exp 6 localised the model's failure to
   per-class OPERATING POINTS (Support Devices 578 calls / 61 true; Atelectasis
   5 / 144 at AUROC .82), while a trained linear readout of the same features
   reaches reliable-11 F1 .514 vs .322. A free-text target cannot express 13
   separate operating points and is dominated by the 96% of decisions that are
   "no". Here the answer is 13 positionally aligned yes/no tokens, the loss is
   weighted per finding so each class's positives count, and at evaluation each
   slot's P(yes) is read directly and thresholded per class on the DEV split --
   the structure that makes the probe work, but with the model's weights adapted.
"""
import numpy as np

import mlx.core as mx
import mlx.nn as nn

from src.eval_accuracy import _truth_positive_set, propagate_hierarchy
from src.labeling import CHEXPERT_CLASSES

FINDABLE = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
N_SLOTS = len(FINDABLE)
YES_ID, NO_ID, NL_ID = 4443, 1904, 107      # single Gemma tokens: "yes", "no", "\n"

INSTRUCTION = (
    "You are a radiologist reading a chest radiograph. For each finding below, in "
    "order, answer yes if it is present or no if it is absent. Give exactly one "
    "answer per line and no other text.\n"
    + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(FINDABLE))
)


def truth_set(row):
    """Scored truth: CheXbert lblcx_, uncertain=ones, hierarchy-propagated --
    the exact convention every number in the paper is scored under, so the
    model is trained on the quantity it is evaluated on."""
    return {c for c in propagate_hierarchy(_truth_positive_set(row, "ones", "lblcx_"))
            if c != "No Finding"}


def target_text(row):
    pos = truth_set(row)
    return "\n".join("yes" if c in pos else "no" for c in FINDABLE)


def class_weights(records, cap):
    """Per-finding weight on a POSITIVE answer = n_neg / n_pos, capped. With it
    the weighted loss is balanced per class (a missed Pneumonia costs what all the
    correct Pneumonia 'no's are worth together) instead of being 96% 'no'."""
    Y = np.array([r["_y"] for r in records], dtype=np.float32)
    npos = Y.sum(0)
    w = np.minimum(cap, (len(Y) - npos) / np.maximum(npos, 1))
    return w.astype(np.float32), npos.astype(int)


def check_tokens(tokenizer):
    for s, i in (("yes", YES_ID), ("no", NO_ID), ("\n", NL_ID)):
        got = tokenizer.encode(s, add_special_tokens=False)
        assert got == [i], f"token id drift for {s!r}: {got} != [{i}]"


def causal_logits(model, input_ids, pixel_values):
    """Forward pass with the model's own causal masking (see module docstring).
    Passing mask=None to model() directly crashes in prepare_inputs_for_multimodal,
    so take embeddings first and run the language model with mask=None."""
    emb = model.get_input_embeddings(input_ids, pixel_values,
                                     mask=mx.ones_like(input_ids)).inputs_embeds
    return model.language_model(input_ids, inputs_embeds=emb).logits.astype(mx.float32)


def decision_weights(labels_np, cmask_np, w_pos, w_neg=1.0, fmt_w=0.05):
    """Per-position loss weight. The k-th yes/no label inside the completion is
    finding k; it gets w_pos[k] if the answer is yes, w_neg if no. Separator and
    end-of-turn tokens get a small fmt_w so the format is still learned but
    cannot dominate. Asserts exactly N_SLOTS decisions per row -- a mis-set mask
    fails loudly here instead of silently training on the wrong tokens."""
    W = np.zeros(labels_np.shape, np.float32)
    for r in range(labels_np.shape[0]):
        pos = np.where(cmask_np[r] > 0)[0]
        W[r, pos] = fmt_w
        dec = [p for p in pos if labels_np[r, p] in (YES_ID, NO_ID)]
        assert len(dec) == N_SLOTS, f"row {r}: {len(dec)} decision tokens, expected {N_SLOTS}"
        for k, p in enumerate(dec):
            W[r, p] = w_pos[k] if labels_np[r, p] == YES_ID else w_neg
    return W


def make_loss_fn(w_pos, norm, fmt_w=0.05):
    """Drop-in replacement for sft_trainer.vision_language_loss_fn: causal, and
    weighted per finding. Normalised by a FIXED constant (mean total weight per
    training example), not per batch, so it is true weighted ERM -- per-batch
    normalisation would shrink exactly the rare-positive examples we upweight."""
    def loss_fn(model, batch, train_on_completions=True, assistant_id=None):
        ids, am = batch["input_ids"], batch["attention_mask"]
        x, labels = ids[:, :-1], ids[:, 1:]
        logits = causal_logits(model, x, batch["pixel_values"])
        cmask = np.array(batch["completion_mask"][:, 1:] * am[:, 1:])
        W = mx.array(decision_weights(np.array(labels), cmask, w_pos, fmt_w=fmt_w))
        ce = nn.losses.cross_entropy(logits, labels)
        return (ce * W).sum() / (norm * ids.shape[0])
    return loss_fn


def slot_probs(model, prompt_ids, pixel_values):
    """Read P(yes) for each of the 13 slots, decoding greedily between {yes,no}
    with a KV cache. Returns (p_yes[13], top1_is_answer[13], answer_mass[13]):
    the last two say whether the model itself would have emitted a yes/no token
    there at all, i.e. whether it has learned the format."""
    cache = model.language_model.make_cache()
    emb = model.get_input_embeddings(prompt_ids, pixel_values,
                                     mask=mx.ones_like(prompt_ids)).inputs_embeds
    logits = model.language_model(prompt_ids, inputs_embeds=emb, cache=cache).logits[:, -1]
    p, top1, mass = [], [], []
    for k in range(N_SLOTS):
        lp = nn.log_softmax(logits.astype(mx.float32), axis=-1)[0]
        ly, ln = lp[YES_ID], lp[NO_ID]
        mx.eval(ly, ln)
        p.append(float(mx.sigmoid(ly - ln)))
        top1.append(int(mx.argmax(lp)) in (YES_ID, NO_ID))
        mass.append(float(mx.exp(ly) + mx.exp(ln)))
        tok = YES_ID if p[-1] >= 0.5 else NO_ID
        nxt = [tok, NL_ID] if k < N_SLOTS - 1 else [tok]
        logits = model.language_model(mx.array([nxt]), cache=cache).logits[:, -1]
    return np.array(p), np.array(top1), np.array(mass)
