"""Per-finding readout fine-tune of MedGemma-4B (LoRA), with a CAUSAL training loss.

Targets the two causes of every earlier fine-tune failure (full account in
src/readout_ft.py): the stock mlx_vlm loss trains with no causal mask (the model
could see each answer token while predicting it), and the free-text objective
could not express the per-class operating points Exp 6 showed are the actual
defect. Four changes, each aimed at one cause:

  causal loss        loss_fn=make_loss_fn(...) replaces the stock loss
  compact answer     13 positionally aligned yes/no tokens, nothing else
  per-class weights  positive answer for finding k weighted n_neg/n_pos (capped)
  oversampling       studies with a rare finding (<100 train positives) repeated

Data: the 2,520 studies in neither test_manifest nor dev_manifest (same split as
every probe). Dev is used ONLY by script 57 to choose per-class thresholds; test
is touched once.

Run the probe first, then read its 57 report before any full run:
  python3 scripts/56_finetune_readout.py --iters 30 --out results/adapters/readout_probe

Gentle variant (after the all-block lr 1e-4 probe flattened P(yes)):
  python3 scripts/56_finetune_readout.py --iters 600 --learning-rate 1e-5 \
      --lora-last 8 --save-every 100 --out results/adapters/readout_gentle
"""
import argparse
import json
import os
import random

import mlx.optimizers as optim
import numpy as np
import pandas as pd
from _bootstrap import load_cfg

# NB: trainer.* directly, NOT mlx_vlm.lora (its `datasets` import hangs here).
from mlx_vlm.utils import load
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, iterate_batches, train
from mlx_vlm.trainer.utils import (find_all_linear_names, get_peft_model,
                                   print_trainable_parameters)

from src.readout_ft import (FINDABLE, INSTRUCTION, N_SLOTS, check_tokens, class_weights,
                            decision_weights, make_loss_fn, target_text, truth_set)


def build_records(cfg, seed, oversample, rare_below):
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    held = set(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid) | \
        set(pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid)
    pool = s[~s.uid.isin(held)]
    recs = []
    for _, r in pool.iterrows():
        paths = [p for p in (eval(r["frontal_paths"]) if isinstance(r["frontal_paths"], str)
                             else []) if p and os.path.exists(p)]
        if not paths:
            continue
        pos = truth_set(r)
        recs.append({"image": paths[0], "uid": r["uid"],
                     "_y": [1 if c in pos else 0 for c in FINDABLE],
                     "messages": [{"role": "user", "content": INSTRUCTION},
                                  {"role": "assistant", "content": target_text(r)}]})
    npos = np.array([r["_y"] for r in recs]).sum(0)
    rare = {k for k in range(N_SLOTS) if npos[k] < rare_below}
    if oversample > 1:
        extra = [r for r in recs if any(r["_y"][k] for k in rare)]
        recs = recs + extra * (oversample - 1)
        print(f"  oversample x{oversample}: +{len(extra) * (oversample - 1)} copies of "
              f"{len(extra)} studies with a rare finding "
              f"({', '.join(FINDABLE[k] for k in sorted(rare))})", flush=True)
    random.Random(seed).shuffle(recs)
    return recs


def verify_mask(train_ds, w_pos, processor):
    """Decode exactly what the loss will train on for one example. The old runs
    were misdiagnosed twice for want of this check."""
    b = next(iterate_batches(train_ds, 1, 2048, train=False))
    labels = np.array(b["input_ids"][:, 1:])
    cm = np.array(b["completion_mask"][:, 1:] * b["attention_mask"][:, 1:])
    W = decision_weights(labels, cm, w_pos)
    idx = np.where(cm[0] > 0)[0]
    toks = processor.tokenizer.convert_ids_to_tokens(labels[0, idx].tolist())
    print(f"  mask check: {len(idx)} trained tokens; decisions/weights = "
          + " ".join(f"{t}:{w:.2g}" for t, w in zip(toks, W[0, idx])), flush=True)


def restrict_lora_to_last(model, n_last):
    """Keep LoRA trainable only in the last `n_last` text blocks.

    The 30-iter probe with LoRA on all 34 blocks at lr 1e-4 overwrote the base
    model's image-dependent signal (P(yes) went flat across studies). Exp 6c found
    the concept is still linearly decodable at the final position, so the defect
    sits at the readout end of the stack; adapting only the last blocks moves the
    operating point while leaving the representation they read from untouched.
    Same re-freeze-by-name approach as 54_train_arm.apply_early_lora (bare module
    names from find_all_linear_names would otherwise match every block). Untrained
    LoRA layers are exact no-ops: lora_b is initialised to zeros."""
    n_blocks = len(model.language_model.model.layers)
    first = n_blocks - n_last
    kept = dropped = 0
    for name, mod in model.language_model.named_modules():
        if ".layers." not in name or not hasattr(mod, "lora_a"):
            continue
        idx = int(name.split(".layers.")[1].split(".")[0])
        if idx < first:
            mod.freeze(); dropped += 1
        else:
            kept += 1
    print(f"  LoRA limited to blocks {first}-{n_blocks - 1} "
          f"({kept} LoRA modules trainable, {dropped} re-frozen)", flush=True)
    assert kept > 0
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/medgemma-4b-it-8bit")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16)
    ap.add_argument("--weight-cap", type=float, default=20.0)
    ap.add_argument("--fmt-weight", type=float, default=0.05)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--rare-below", type=int, default=100)
    ap.add_argument("--lora-last", type=int, default=0,
                    help="LoRA only on the last N text blocks (0 = all blocks)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="checkpoint interval (0 = iters//4, min 50)")
    ap.add_argument("--out", default="results/adapters/readout_probe")
    args = ap.parse_args()

    cfg = load_cfg("configs/config_medgemma_estr.yaml")
    seed = cfg.get("seed", 20260705)
    print("Building training set (train pool = not test, not dev)...", flush=True)
    recs = build_records(cfg, seed, args.oversample, args.rare_below)
    w_pos, npos = class_weights(recs, args.weight_cap)
    # fixed normaliser = mean total loss weight per example (see make_loss_fn)
    Y = np.array([r["_y"] for r in recs])
    norm = float((Y * w_pos + (1 - Y)).sum(1).mean() + args.fmt_weight * (N_SLOTS + 1))
    print(f"  {len(recs)} examples, {Y.mean():.3%} positive cells")
    for c, n, w in zip(FINDABLE, npos, w_pos):
        print(f"    {c:<28} pos={n:<5} w_pos={w:.2f}")
    print(f"  loss normaliser {norm:.2f}", flush=True)

    model, processor = load(args.model, processor_config={"trust_remote_code": True})
    check_tokens(processor.tokenizer)
    train_ds = VisionDataset(recs, model.config.__dict__, processor,
                             train_on_completions=True)
    verify_mask(train_ds, w_pos, processor)

    modules = find_all_linear_names(model.language_model)
    model = get_peft_model(model, modules, rank=args.lora_rank, alpha=args.lora_alpha,
                           dropout=0.0, verbose=False)
    if args.lora_last:
        model = restrict_lora_to_last(model, args.lora_last)
    print_trainable_parameters(model)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "readout_meta.json"), "w") as f:
        json.dump({**vars(args), "w_pos": w_pos.tolist(), "train_pos": npos.tolist(),
                   "norm": norm, "classes": FINDABLE, "instruction": INSTRUCTION,
                   "n_examples": len(recs)}, f, indent=2)
    adapter_file = os.path.join(args.out, "adapters.safetensors")
    targs = TrainingArgs(
        batch_size=1, iters=args.iters, steps_per_report=5 if args.iters <= 100 else 25,
        steps_per_eval=10_000_000, steps_per_save=args.save_every or max(50, args.iters // 4),
        val_batches=0, max_seq_length=2048, adapter_file=adapter_file,
        grad_checkpoint=True, learning_rate=args.learning_rate,
        grad_clip=1.0, gradient_accumulation_steps=1, full_finetune=False)
    print(f"Training: {args.iters} iters, lr={args.learning_rate}, causal weighted loss",
          flush=True)
    train(model=model, optimizer=optim.Adam(learning_rate=args.learning_rate),
          train_dataset=train_ds, val_dataset=None, args=targs,
          loss_fn=make_loss_fn(w_pos, norm, args.fmt_weight), train_on_completions=True)
    print(f"Done. Adapter -> {adapter_file}")


if __name__ == "__main__":
    main()
