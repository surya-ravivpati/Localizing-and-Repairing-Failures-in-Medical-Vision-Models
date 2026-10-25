"""Three-arm alignment intervention: training side.

The mentor's design, run on the same 2,520 / 552 / 754 split as every probe:

  baseline   encoder frozen, connector frozen, LLM frozen   (= current MedGemma)
  connector  connector TRAINABLE, everything else frozen    (concept extraction)
  both       connector trainable + LoRA on the early LLM     (token utilisation)
             blocks that ingest the visual tokens

Baseline needs no training and is already on disk as the Stage-A run.

A NOTE ON WHAT THIS IS TESTING. Section 9.8 probed the model's own hidden states
and found the visual signal flat across all 34 text blocks, with 64% of the
probe-recovered misses still linearly decodable from the vector generation runs
from. On that evidence the failure is expression, not attention or extraction, and
both trained arms act upstream of it. The experiment is run anyway because it is
properly controlled: a null here is positive evidence for the §9.8 reading, and a
gain would falsify it. Either outcome is worth the compute; what would not be
worth it is assuming the answer.

Two hard-won settings are non-negotiable and are asserted at startup:
  * train_on_completions=True -- with it False the loss is averaged over the whole
    sequence, which is dominated by an instruction identical in every example, and
    training collapses (three runs lost to this).
  * --normal-ratio -- the train pool is 64.9% normal; leaving it unbalanced lets
    the majority string win.

Usage
  python3 scripts/54_train_arm.py --arm connector --iters 60  --out results/arms/_probe
  python3 scripts/54_train_arm.py --arm connector --iters 800 --out results/arms/connector
"""
import argparse
import os
import random

import mlx.optimizers as optim
from mlx.utils import tree_flatten
from _bootstrap import load_cfg

from mlx_vlm.utils import load
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.trainer.utils import (find_all_linear_names, get_peft_model,
                                   print_trainable_parameters)

import ast

import pandas as pd

from src import prompts
from src.eval_accuracy import _truth_positive_set
from src.labeling import CHEXPERT_CLASSES

FINDABLE = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
NORMAL = "No acute cardiopulmonary finding"


def target_text(row):
    """CheXbert-positive findings as canonical names, recoverable by normalize_dx."""
    pos = [c for c in FINDABLE if row.get(f"lblcx_{c}") == 1.0]
    return "; ".join(pos) if pos else NORMAL


def build_records(cfg, seed, normal_ratio, condition="perception_only"):
    """Training records that use the SAME prompt the baseline ran.

    Parity matters more than it looks. The arms are scored against MedGemma's own
    Stage-A run, and against the 621 misses and 129 recoveries derived from it,
    all of which used the free-text `perception_only` scaffold. Training on a
    different instruction would make the conversion metric meaningless — and the
    detection instruction in particular is known to drive this model to answer
    "normal" on nearly every study, so a baseline built on it would be degenerate.

    normal_ratio caps normal studies at that multiple of the abnormal count; the
    pool is 64.9% normal and an unbalanced run lets the majority string win.
    """
    out = cfg["paths"]["out_dir"]
    s = pd.read_csv(os.path.join(out, "studies_labeled.csv"))
    test = set(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid)
    dev = set(pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid)
    pool = s[~s.uid.isin(test | dev)]
    recs = []
    for _, r in pool.iterrows():
        raw = r.get("frontal_paths")
        paths = ast.literal_eval(raw) if isinstance(raw, str) and raw.strip() else []
        paths = [p for p in paths if p and os.path.exists(p)]
        if not paths:
            continue
        system, user = prompts.build_freetext(condition, r.get("indication", ""))
        recs.append({"image": paths[0],
                     "messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": user},
                                  {"role": "assistant", "content": target_text(r)}]})
    rng = random.Random(seed)
    rng.shuffle(recs)
    if normal_ratio is not None:
        abn = [x for x in recs if x["messages"][-1]["content"] != NORMAL]
        nrm = [x for x in recs if x["messages"][-1]["content"] == NORMAL]
        keep = int(round(normal_ratio * len(abn)))
        recs = abn + nrm[:keep]
        rng.shuffle(recs)
        print(f"  balanced: {len(abn)} abnormal + {min(keep, len(nrm))} normal")
    return recs


def apply_early_lora(model, n_blocks, rank, alpha):
    """LoRA restricted to the first `n_blocks` text blocks.

    `find_all_linear_names` returns bare module names, so `get_peft_model` would
    match every block. Restricting to the blocks that first ingest visual tokens
    is the whole point of arm `both`, so the later blocks are re-frozen by name
    afterwards rather than never adapted.
    """
    modules = find_all_linear_names(model.language_model)
    model = get_peft_model(model, modules, rank=rank, alpha=alpha,
                           dropout=0.0, verbose=False)
    kept, dropped = 0, 0
    for name, mod in model.language_model.named_modules():
        if ".layers." not in name:
            continue
        try:
            idx = int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if idx >= n_blocks and hasattr(mod, "freeze"):
            mod.freeze(); dropped += 1
        elif idx < n_blocks:
            kept += 1
    print(f"  LoRA limited to blocks 0-{n_blocks-1} "
          f"({kept} modules adapted, {dropped} re-frozen)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["connector", "both"])
    ap.add_argument("--model", default="mlx-community/medgemma-4b-it-8bit")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16)
    ap.add_argument("--lora-blocks", type=int, default=8,
                    help="arm `both`: adapt only the first N of 34 text blocks")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--resize", type=int, default=448)
    ap.add_argument("--normal-ratio", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--condition", default="perception_only",
                    help="MUST match the baseline Stage-A run being compared against")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_cfg("configs/config_medgemma_estr.yaml")
    ds = build_records(cfg, cfg.get("seed", 20260705), args.normal_ratio,
                       condition=args.condition)
    print(f"[arm:{args.arm}] {len(ds)} training examples")

    model, processor = load(args.model, processor_config={"trust_remote_code": True})
    train_ds = VisionDataset(ds, model.config.__dict__, processor,
                             image_resize_shape=[args.resize, args.resize],
                             train_on_completions=True)

    model.freeze()
    model.multi_modal_projector.unfreeze()
    if args.arm == "both":
        model = apply_early_lora(model, args.lora_blocks, args.lora_rank, args.lora_alpha)
        model.multi_modal_projector.unfreeze()   # get_peft_model re-freezes

    trainable = tree_flatten(model.trainable_parameters())
    n = sum(v.size for _, v in trainable)
    heads = sorted({k.split(".layers.")[0] if ".layers." in k else k.rsplit(".", 1)[0]
                    for k, _ in trainable})
    print(f"[arm:{args.arm}] trainable {n/1e6:.3f}M params across {len(trainable)} tensors")
    print(f"  groups: {heads[:6]}{' …' if len(heads) > 6 else ''}")
    if n == 0:
        raise SystemExit("nothing trainable — freeze/unfreeze did not take")
    if not any("multi_modal_projector" in k for k, _ in trainable):
        raise SystemExit("connector is not trainable — both arms require it")

    os.makedirs(args.out, exist_ok=True)
    weights_file = os.path.join(args.out, "adapters.safetensors")
    targs = TrainingArgs(
        batch_size=args.batch_size, iters=args.iters, steps_per_report=10,
        steps_per_eval=10_000_000, steps_per_save=max(50, args.iters // 4),
        val_batches=0, max_seq_length=args.max_seq_length,
        adapter_file=weights_file, grad_checkpoint=args.grad_checkpoint,
        learning_rate=args.learning_rate, grad_clip=1.0,
        gradient_accumulation_steps=1, full_finetune=False,
    )
    print(f"[arm:{args.arm}] {args.iters} iters, lr={args.learning_rate}, "
          f"bs={args.batch_size}")
    train(model=model, optimizer=optim.Adam(learning_rate=args.learning_rate),
          train_dataset=train_ds, val_dataset=None, args=targs,
          train_on_completions=True)
    print(f"[arm:{args.arm}] done -> {weights_file}")
    print("[arm] NEXT: eval with 55_eval_arm.py and CHECK RAW GENERATIONS before "
          "believing any metric — a broken adapter and a collapsed one score alike.")


if __name__ == "__main__":
    main()
