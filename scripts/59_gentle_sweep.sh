#!/bin/zsh
# Gentle readout fine-tune: lr 1e-5, LoRA on the last 8 of 34 text blocks, 600 iters,
# checkpoint every 100. Each checkpoint is scored on the 80-study dev SELECTION set.
#
# DECISION RULE (fixed before any result is seen):
#   select  = checkpoint with the highest pooled-cell AUROC on the selection set
#   accept  only if, on a DISJOINT 120-study dev confirmation set scored once, the
#             selected checkpoint beats the base model: paired study-bootstrap
#             delta > 0 with the 95% CI excluding 0. (The base model's 0.767 on the
#             selection set is the bar to clear there; confirmation guards against
#             picking the luckiest of six checkpoints.)
#   reject  otherwise -> no full run; report as a null.
set -e
cd "/Users/suryaravipati/Desktop/Research Proj's/independent/pipeline"
OUT=results/adapters/readout_gentle
LOG=results/readout_gentle.log

python3 scripts/56_finetune_readout.py --iters 600 --learning-rate 1e-5 \
    --lora-last 8 --save-every 100 --out $OUT >> $LOG 2>&1

for it in 0000100 0000200 0000300 0000400 0000500 0000600; do
  echo "=== EVAL checkpoint $it ===" >> $LOG
  python3 scripts/57_eval_readout.py --mode probe --set probe --adapter $OUT \
      --weights $OUT/${it}_adapters.safetensors --tag gentle_$it >> $LOG 2>&1
  python3 scripts/58_compare_readout_probe.py --a base --b gentle_$it 2>&1 \
      | grep -E "pooled-cell|yes-rate|distinct" >> $LOG
done
echo "SWEEP DONE" >> $LOG
