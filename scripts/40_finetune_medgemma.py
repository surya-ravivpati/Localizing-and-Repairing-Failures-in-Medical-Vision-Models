"""LoRA fine-tune MedGemma-4B on IU X-Ray train split for 13-class finding detection.

The two-stage experiment localized the score ceiling to PERCEPTION, and every
zero-shot lever (bigger model, full-res, structure, ensembling, cross-model) failed
to move 13-class F1. Fine-tuning is the one legitimate lever left: teach the model
the finding distribution from labeled TRAIN studies (never test/dev), then measure
13-class F1 on the locked test set with the same scorer as everywhere else.

Data: the 2,520 studies NOT in test_manifest/dev_manifest. Target = the CheXbert
(lblcx_) present findings, phrased with canonical class names so the existing
normalize_dx recovers them 1:1 at eval. Question = a fixed detection instruction
listing the label space (so train and eval prompts match exactly).

Reuses MLX-VLM's trainer internals (get_peft_model + VisionDataset + train) but
builds the dataset from our data directly — HF load_dataset can't cleanly ingest
local image+label pairs. Smoke-test first:  --limit 8 --iters 2

Requires:  pip install 'mlx-vlm[train]'   (datasets etc.)
"""
import argparse
import os
import random

import mlx.optimizers as optim
import pandas as pd
from _bootstrap import load_cfg

from src.labeling import CHEXPERT_CLASSES

# NB: import from trainer.* directly, NOT mlx_vlm.lora — lora.py's first line is
# `from datasets import load_dataset`, and the HF datasets/pyarrow stack hangs on
# import on this machine. trainer.utils / sft_trainer are datasets-free.
from mlx_vlm.utils import load
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.trainer.utils import (find_all_linear_names, get_peft_model,
                                   print_trainable_parameters)

FINDABLE = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
INSTRUCTION = (
    "You are a radiologist reading a chest radiograph. Decide which of the following "
    "findings are present: " + ", ".join(FINDABLE) + ". "
    "Reply with only the present findings by name, separated by semicolons, or "
    "'No acute cardiopulmonary finding' if none are present."
)


def target_text(row) -> str:
    """CheXbert-positive findings as canonical names (recoverable by normalize_dx)."""
    pos = [c for c in FINDABLE if row.get(f"lblcx_{c}") == 1.0]
    return "; ".join(pos) if pos else "No acute cardiopulmonary finding"


def build_records(cfg, limit, seed, normal_ratio=None):
    """Plain list of {image-path, pre-formatted messages}; no HF `datasets` dep
    (its pyarrow stack hangs on import here). VisionDataset only needs len()+[idx],
    and mlx_vlm loads image paths lazily per item — so memory stays flat.

    `normal_ratio` caps normal studies at that multiple of the abnormal count.
    The train pool is 64.9% normal (1635 of 2520), and an unbalanced run COLLAPSES:
    a 1500-iteration fine-tune plateaued at loss 1.82 by iteration 200 and then
    emitted the single string "No acute cardiopulmonary finding" for 754/754 test
    studies, scoring F1 0.000. Always saying the majority target is very nearly
    loss-optimal when the majority target is two thirds of the data, so the model
    never learns to name a finding at all. normal_ratio=1.0 gives a balanced set.
    """
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    test = set(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid)
    dev = set(pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid)
    pool = s[~s.uid.isin(test | dev)].copy()
    pool["frontal_paths"] = pool["frontal_paths"].map(
        lambda x: eval(x) if isinstance(x, str) else x)
    recs = []
    for _, r in pool.iterrows():
        paths = [p for p in (r["frontal_paths"] or []) if p and os.path.exists(p)]
        if not paths:
            continue
        recs.append({
            "image": paths[0],
            "messages": [{"role": "user", "content": INSTRUCTION},
                         {"role": "assistant", "content": target_text(r)}],
        })
    rng = random.Random(seed)
    rng.shuffle(recs)
    if normal_ratio is not None:
        NORMAL = "No acute cardiopulmonary finding"
        abn = [r for r in recs if r["messages"][1]["content"] != NORMAL]
        nrm = [r for r in recs if r["messages"][1]["content"] == NORMAL]
        keep = int(round(normal_ratio * len(abn)))
        recs = abn + nrm[:keep]
        rng.shuffle(recs)
        print(f"  balanced: {len(abn)} abnormal + {min(keep, len(nrm))} normal "
              f"(pool had {len(nrm)} normal)", flush=True)
    return recs[:limit] if limit else recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/medgemma-4b-it-8bit")
    ap.add_argument("--limit", type=int, default=0, help="cap #train studies (0=all 2520)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--resize", type=int, default=0, help="square image resize (0=native)")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--normal-ratio", type=float, default=None,
                    help="cap normal studies at this multiple of the abnormal count "
                         "(1.0 = balanced). Unset reproduces the collapsed run.")
    ap.add_argument("--out", default="results/adapters/medgemma_det")
    args = ap.parse_args()

    cfg = load_cfg("configs/config_medgemma_estr.yaml")
    print("Building training set (train pool = studies not in test/dev)...", flush=True)
    ds = build_records(cfg, args.limit, cfg.get("seed", 20260705),
                       normal_ratio=args.normal_ratio)
    print(f"  {len(ds)} training examples | e.g. target: "
          f"{ds[0]['messages'][1]['content']!r}", flush=True)

    print(f"Loading {args.model} ...", flush=True)
    model, processor = load(args.model, processor_config={"trust_remote_code": True})
    config = model.config.__dict__

    resize = [args.resize, args.resize] if args.resize else None
    # train_on_completions=True is ESSENTIAL, not a tuning choice. With it False
    # the loss is averaged over the WHOLE sequence -- 256 image tokens plus an
    # instruction that lists all 13 finding names and is byte-identical in every
    # example -- so the few tokens we actually want to learn contribute almost
    # nothing to the gradient. Two runs proved it: loss converged to ~1.82 by
    # iteration 200 and then sat there, and rebalancing the classes 65/35 -> 50/50
    # moved it by 0.0009 (1.8203 -> 1.8194), because the class signal was never
    # what was being optimised. Setting it True masks the loss to the assistant
    # response. VisionDataset then emits a real completion_mask, so sft_trainer
    # takes the `"completion_mask" in batch` branch rather than its hardcoded
    # assistant_id=77091 fallback.
    train_ds = VisionDataset(ds, config, processor, image_resize_shape=resize,
                             train_on_completions=True)

    # LoRA on the language model only (vision stack frozen); this teaches the
    # finding-naming distribution without disturbing perception weights. Inlined
    # from lora.setup_model_for_training to avoid importing the datasets-dependent
    # lora module.
    print("Applying LoRA to language model...", flush=True)
    modules = find_all_linear_names(model.language_model)
    model = get_peft_model(model, modules, rank=args.lora_rank,
                           alpha=args.lora_alpha, dropout=0.0, verbose=False)
    print_trainable_parameters(model)

    os.makedirs(args.out, exist_ok=True)
    adapter_file = os.path.join(args.out, "adapters.safetensors")
    targs = TrainingArgs(
        batch_size=args.batch_size, iters=args.iters, steps_per_report=10,
        steps_per_eval=10_000_000, steps_per_save=max(50, args.iters // 4),
        val_batches=0, max_seq_length=args.max_seq_length, adapter_file=adapter_file,
        grad_checkpoint=args.grad_checkpoint, learning_rate=args.learning_rate,
        grad_clip=1.0, gradient_accumulation_steps=1, full_finetune=False,
    )
    print(f"Training LoRA: {args.iters} iters, bs={args.batch_size}, lr={args.learning_rate}")
    train(model=model, optimizer=optim.Adam(learning_rate=args.learning_rate),
          train_dataset=train_ds, val_dataset=None, args=targs,
          train_on_completions=True)   # must match VisionDataset above
    print(f"Done. Adapter -> {adapter_file}")


if __name__ == "__main__":
    main()
