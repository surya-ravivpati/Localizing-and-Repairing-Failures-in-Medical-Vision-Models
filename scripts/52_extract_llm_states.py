"""Experiment 6c: extend the probe INSIDE the language model.

Everything in section 9 stops at the vision-language interface -- the 256 vectors
handed to the language model. That localises the loss to "after the projector",
which is not yet a mechanism. Three failure modes remain, and they need different
fixes:

  A  the information is present but the language model does not attend to the
     right visual tokens
  B  it is attended to, but the mapping from visual features to clinical concepts
     is weak
  C  the concept is internally recoverable, but generation fails to express it

The interface probe cannot separate these, because it never looks inside the
language model. This does, by probing the model's own hidden states at two places
per layer:

  img_L{k}    mean ++ max over the 256 IMAGE-token positions after block k
              -- what the language model has made of the picture
  last_L{k}   the FINAL position after block k -- the vector the model is about to
              generate from, i.e. what is available at the decision point

The discrimination follows directly. If a finding is linearly recoverable from
`last_L{k}` at some depth but the model does not say it, the concept reached the
decision point and generation failed to express it: mode C, and attention tuning
would be aimed at the wrong stage. If it is present at the interface but decays
inside the stack and is NOT recoverable at `last`, the language model is failing to
carry it forward: mode A or B, and connector/attention interventions are indicated.

The prompt is byte-identical to the one Stage A ran (`perception_only`, free-text),
so these states are the ones that actually produced the 621 misses the section
tracks. Nothing is generated -- this is a single forward pass per study.

Resumable: one .npz per study, skipped if present.

Usage
  python3 scripts/52_extract_llm_states.py --split test --limit 24 --out-suffix pilot
  python3 scripts/52_extract_llm_states.py --split all
"""
import argparse
import ast
import os
import sys
import time

import numpy as np
import pandas as pd
from _bootstrap import load_cfg

from src import prompts

# Gemma3-4B text stack is 34 blocks. Sample the entry, the quarters and the exit.
DEFAULT_LAYERS = [0, 6, 12, 18, 24, 30, 33]
IMAGE_TOKEN_INDEX = 262144


def resolve_split(out_dir, split):
    s = pd.read_csv(os.path.join(out_dir, "studies_labeled.csv"), usecols=["uid"])
    test = set(pd.read_csv(os.path.join(out_dir, "test_manifest.csv")).uid)
    dev = set(pd.read_csv(os.path.join(out_dir, "dev_manifest.csv")).uid)
    if split == "test":
        return test
    if split == "dev":
        return dev
    if split == "train":
        return set(s.uid) - test - dev
    if split == "all":
        return set(s.uid)
    raise SystemExit(f"unknown split {split!r}")


def image_for(row):
    """First readable frontal, else lateral -- the same rule Stage A and the
    vision-side extraction both use, so the three are on identical inputs."""
    for col in ("frontal_paths", "lateral_paths"):
        raw = row.get(col)
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            paths = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        for p in paths:
            if p and os.path.exists(p):
                return p
    return None


def pool(tokens):
    return np.concatenate([tokens.mean(axis=0), tokens.max(axis=0)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/medgemma-4b-it-8bit")
    ap.add_argument("--split", default="all", choices=["train", "dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--condition", default="perception_only",
                    help="must match the Stage-A run whose misses are being tracked")
    ap.add_argument("--out-suffix", default="llm8bit")
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    feat_dir = os.path.join(out, "exp6_features", args.out_suffix)
    os.makedirs(feat_dir, exist_ok=True)

    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    keep = resolve_split(out, args.split)
    studies = studies[studies.uid.isin(keep)].reset_index(drop=True)
    if args.limit:
        studies = studies.head(args.limit)
    todo = [r for _, r in studies.iterrows()
            if not os.path.exists(os.path.join(feat_dir, f"{r.uid}.npz"))]
    print(f"[exp6c] {len(studies)} studies | layers={layers} -> {feat_dir}")
    print(f"[exp6c] {len(studies)-len(todo)} already done, {len(todo)} to extract")
    if not todo:
        print("[exp6c] nothing to do"); return

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    t0 = time.time()
    model, processor = load(args.model, processor_config={"trust_remote_code": True})
    lm = model.language_model.model            # Gemma3Model: .layers, .norm
    print(f"[exp6c] loaded in {time.time()-t0:.1f}s | {len(lm.layers)} text blocks")
    if max(layers) >= len(lm.layers):
        raise SystemExit(f"layer index out of range (0..{len(lm.layers)-1})")

    from mlx_vlm.models.base import create_attention_mask

    t0, done, skipped = time.time(), 0, []
    for i, row in enumerate(todo):
        path = image_for(row)
        if path is None:
            skipped.append(int(row.uid)); continue
        system, user = prompts.build_freetext(args.condition, row.get("indication", ""))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        prompt = apply_chat_template(processor, model.config, messages, num_images=1)
        inputs = prepare_inputs(processor, images=[path], prompts=[prompt],
                                image_token_index=IMAGE_TOKEN_INDEX)
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        mask_in = inputs.get("attention_mask")

        feats = model.get_input_embeddings(input_ids, pixel_values, mask_in)
        h = feats.inputs_embeds
        attn = feats.attention_mask_4d
        h = h * mx.array(lm.config.hidden_size ** 0.5, mx.bfloat16).astype(h.dtype)

        # replicate Gemma3Model.__call__ so intermediate states can be captured;
        # the sliding-window/global alternation must be reproduced exactly or the
        # states are not the ones the model actually computes.
        cache = [None] * len(lm.layers)
        if attn is None:
            global_mask = create_attention_mask(h, cache[lm.sliding_window_pattern - 1])
            sliding = (create_attention_mask(h, cache[0], window_size=lm.window_size)
                       if lm.sliding_window_pattern > 1 else None)
        rec = {}
        img_pos = np.where(np.array(input_ids)[0] == IMAGE_TOKEN_INDEX)[0]
        for k, (layer, c) in enumerate(zip(lm.layers, cache)):
            local = attn
            if attn is None:
                is_global = k % lm.sliding_window_pattern == lm.sliding_window_pattern - 1
                local = global_mask if is_global else sliding
            h = layer(h, local, c)
            if k in layers:
                # bfloat16 has no numpy equivalent -- cast inside MLX first
                hf = h.astype(mx.float32); mx.eval(hf)
                arr = np.asarray(hf)[0]
                rec[f"img_L{k}"] = (pool(arr[img_pos]) if len(img_pos)
                                    else np.zeros(arr.shape[1] * 2, np.float32))
                rec[f"last_L{k}"] = arr[-1].astype(np.float32)
        hn = lm.norm(h).astype(mx.float32); mx.eval(hn)
        arrn = np.asarray(hn)[0]
        rec["last_final"] = arrn[-1].astype(np.float32)
        rec["n_image_tokens"] = np.array([len(img_pos)], np.int32)
        np.savez_compressed(os.path.join(feat_dir, f"{row.uid}.npz"), **rec)

        done += 1
        if done % 25 == 0 or i == len(todo) - 1:
            rate = (time.time() - t0) / done
            print(f"  {done}/{len(todo)}  {rate:.2f}s/study  "
                  f"eta {(len(todo)-i-1)*rate/60:.0f}m", flush=True)

    print(f"[exp6c] extracted {done} in {(time.time()-t0)/60:.1f} min -> {feat_dir}")
    if skipped:
        print(f"[exp6c] WARNING skipped {len(skipped)} with no readable image")

    files = sorted(f for f in os.listdir(feat_dir) if f.endswith(".npz"))
    if len(files) >= 2:
        sample = [np.load(os.path.join(feat_dir, f)) for f in files[:min(40, len(files))]]
        print("\n[gate] site shapes and between-study variance")
        ok = True
        ntok = int(np.median([s["n_image_tokens"][0] for s in sample]))
        print(f"  image tokens per study (median): {ntok}")
        if ntok == 0:
            print("  FAIL: no image tokens located — check IMAGE_TOKEN_INDEX"); ok = False
        for key in [k for k in sample[0].files if k != "n_image_tokens"]:
            M = np.stack([s[key] for s in sample])
            sd = float(M.std(axis=0).mean())
            flag = "" if sd > 1e-6 else "  <-- CONSTANT"
            if sd <= 1e-6:
                ok = False
            print(f"  {key:>12}  dim={M.shape[1]:>5}  between-study sd={sd:.4g}{flag}")
        print(f"[gate] {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
