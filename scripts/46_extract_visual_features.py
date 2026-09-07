"""Experiment 6, stage 1: extract MedGemma's INTERNAL visual representations.

Runs only the vision pathway — no language model, no generation, no API, no cost.
For every study we record what the model sees at three points along the pathway
from pixels to language:

    image
      |
      +-- [pixels]  32x32 grayscale                      <- CONTROL site
      |
      v  SigLIP vision tower (27 layers, 4096 patches x 1152)
      +-- [layer_k] pooled patch tokens, k = 0..27       <- SITE 1 (encoder)
      |
      v  multi_modal_projector  (4096 -> 256 tokens, 2560-dim)
      +-- [proj]  the vectors literally spliced into the                SITE 2
                  language model's token stream          <- (vision->language iface)

Site 2 is the decisive one. A linear probe that succeeds there while the VLM's own
language head fails proves the information REACHED the language model and was not
used (a readout/alignment failure). Information present at site 1 but absent at
site 2 instead indicts the projector's 4096->256 pooling.

Pooling: each site is reduced to mean-pool ++ max-pool over tokens. Mean alone
washes out focal findings (a small effusion is a few patches out of 4096); max
keeps the strongest local response. Probes get both.

The image fed here is the SAME file MedGemma's Stage A consumed (frontal_paths[0],
falling back to lateral for the studies with no frontal, exactly as Exp 1 did), so
the probe and the language head see byte-identical input. That equivalence is the
whole point of the experiment - do not "improve" the preprocessing here.

Resumable: one .npz per study, skipped if present (mirrors run_inference's resume
semantics). Safe to Ctrl-C and rerun.

Usage
  # pilot (gate) - 50 studies, checks shapes/variance/timing before the full run
  python3 scripts/46_extract_visual_features.py --split test --limit 50 --out-suffix pilot
  # full extraction, 8-bit
  python3 scripts/46_extract_visual_features.py --split all
  # the two controls
  python3 scripts/46_extract_visual_features.py --split all --random-init
  python3 scripts/46_extract_visual_features.py --split all --model mlx-community/medgemma-4b-it-4bit
"""
import argparse
import ast
import os
import sys
import time

import numpy as np
import pandas as pd
from _bootstrap import load_cfg
from PIL import Image

# Layers probed by default: every 3rd of the 27 SigLIP blocks, plus the input
# embedding (0) and the final block (27). hidden_states[0] is the patch+position
# embedding BEFORE any attention; hidden_states[k] is the output of block k.
DEFAULT_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
PIXEL_SIDE = 32          # raw-pixel control resolution


def resolve_split(out_dir: str, split: str) -> set[int]:
    """Locked splits: test/dev come from their manifests, train is the remainder.

    studies_labeled.csv has no `split` column - the manifests ARE the split, and
    they are frozen (the test set has been locked since the main study). The probe
    trains on `train`, tunes on `dev`, and touches `test` exactly once.
    """
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


def image_for(row) -> str | None:
    """The single image MedGemma actually receives: first frontal, else first
    lateral. Identical rule to Exp 1's frontal-only variants, which kept the
    locked n=754 whole rather than dropping the 32 frontal-less studies."""
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


def pool(tokens: np.ndarray) -> np.ndarray:
    """[T, D] -> [2D] = mean-pool ++ max-pool over the token axis."""
    return np.concatenate([tokens.mean(axis=0), tokens.max(axis=0)]).astype(np.float32)


def build_random_encoder(model):
    """Control: same SigLIP architecture, freshly initialised weights.

    Random-feature encoders are a famously strong baseline, so "the probe reads
    pathology off layer k" only means something if the TRAINED encoder beats an
    UNTRAINED one of identical shape and identical pooling.
    """
    from mlx_vlm.models.gemma3.vision import VisionModel
    return VisionModel(model.config.vision_config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/medgemma-4b-it-8bit",
                    help="8-bit is the checkpoint the study ran on; 4-bit is the "
                         "known-degraded-vision control")
    ap.add_argument("--split", default="all", choices=["train", "dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--uids-file", default=None, help="CSV with a uid column")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)),
                    help="comma-separated encoder layers, or 'all' for 0..27")
    ap.add_argument("--random-init", action="store_true",
                    help="control: untrained encoder of identical architecture")
    ap.add_argument("--out-suffix", default=None,
                    help="feature dir suffix (default: derived from model+init)")
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    layers = list(range(28)) if args.layers == "all" else \
        [int(x) for x in args.layers.split(",") if x.strip()]
    if max(layers) > 27 or min(layers) < 0:
        raise SystemExit("layers must be within 0..27")

    tag = args.out_suffix or (
        ("randinit" if args.random_init else args.model.rsplit("-", 1)[-1]))
    feat_dir = os.path.join(out, "exp6_features", tag)
    os.makedirs(feat_dir, exist_ok=True)

    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    keep = resolve_split(out, args.split)
    if args.uids_file:
        keep &= set(pd.read_csv(args.uids_file).uid)
    studies = studies[studies.uid.isin(keep)].reset_index(drop=True)
    if args.limit:
        studies = studies.head(args.limit)
    print(f"[exp6] {len(studies)} studies | split={args.split} | model={args.model}"
          f"{' RANDOM-INIT' if args.random_init else ''}")
    print(f"[exp6] layers={layers} -> {feat_dir}")

    todo = [r for _, r in studies.iterrows()
            if not os.path.exists(os.path.join(feat_dir, f"{r.uid}.npz"))]
    print(f"[exp6] {len(studies) - len(todo)} already done, {len(todo)} to extract")
    if not todo:
        print("[exp6] nothing to do"); return

    import mlx.core as mx
    from mlx_vlm import load
    t0 = time.time()
    model, processor = load(args.model)
    vision = build_random_encoder(model) if args.random_init else model.vision_tower
    projector = model.multi_modal_projector
    image_proc = processor.image_processor
    print(f"[exp6] model loaded in {time.time() - t0:.1f}s")

    t0, done, skipped = time.time(), 0, []
    for i, row in enumerate(todo):
        path = image_for(row)
        if path is None:
            skipped.append(int(row.uid))
            continue
        img = Image.open(path).convert("RGB")
        pv = image_proc(images=[img], return_tensors="np")["pixel_values"]
        # image_processor emits NCHW (torch order); MLX convolutions want NHWC.
        x = mx.array(np.asarray(pv, dtype=np.float32)).transpose(0, 2, 3, 1)

        pooled, _, hidden = vision(x, output_hidden_states=True)
        feats = projector(pooled)          # [1, 256, 2560] -> what the LLM receives
        mx.eval(pooled, feats, *[hidden[k] for k in layers])

        rec = {f"layer_{k}": pool(np.asarray(hidden[k])[0]) for k in layers}
        rec["proj"] = pool(np.asarray(feats)[0])
        rec["pixels"] = (np.asarray(
            img.convert("L").resize((PIXEL_SIDE, PIXEL_SIDE), Image.BILINEAR),
            dtype=np.float32).ravel() / 255.0)
        np.savez_compressed(os.path.join(feat_dir, f"{row.uid}.npz"), **rec)

        done += 1
        if done % 25 == 0 or i == len(todo) - 1:
            rate = (time.time() - t0) / done
            left = (len(todo) - i - 1) * rate
            print(f"  {done}/{len(todo)}  {rate:.2f}s/img  eta {left/60:.0f}m", flush=True)

    print(f"[exp6] extracted {done} in {(time.time()-t0)/60:.1f} min -> {feat_dir}")
    if skipped:
        print(f"[exp6] WARNING skipped {len(skipped)} studies with no readable "
              f"image: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    # --- gate: shapes, and that representations actually vary across studies ---
    files = sorted(f for f in os.listdir(feat_dir) if f.endswith(".npz"))
    if len(files) >= 2:
        sample = [np.load(os.path.join(feat_dir, f)) for f in files[:min(50, len(files))]]
        print("\n[gate] site shapes and between-study variance")
        ok = True
        for key in [f"layer_{k}" for k in layers] + ["proj", "pixels"]:
            M = np.stack([s[key] for s in sample])
            sd = float(M.std(axis=0).mean())
            flag = "" if sd > 1e-6 else "  <-- CONSTANT, probe would be meaningless"
            if sd <= 1e-6:
                ok = False
            print(f"  {key:>10}  dim={M.shape[1]:>5}  between-study sd={sd:.4g}{flag}")
        print(f"[gate] {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
