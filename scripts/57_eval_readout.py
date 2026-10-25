"""Evaluate the per-finding readout fine-tune (script 56), or the base model as control.

For every study it reads P(yes) for each of the 13 slots directly from the model
(src.readout_ft.slot_probs) instead of parsing generated prose, so each finding
gets its own operating point -- the thing Exp 6 showed is broken.

Two modes:

  --mode probe  the go/no-go check after a short run. A dev subset (half abnormal,
                seeded), scored for: does the model emit yes/no at all (format),
                do its P(yes) vary across studies (the collapse signature is that
                they do not), and do they rank positives above negatives (AUROC).
                Always run it for BOTH the adapter and --adapter none: the base
                model's own slot probabilities are the control that separates
                "fine-tuning helped" from "reading probabilities helped".

  --mode full   all of dev (552) + test (754). Per-class thresholds chosen on DEV
                to maximise that class's F1, applied once to TEST, scored on the
                paper's ladder (CheXbert lblcx_, uncertain=ones, hierarchy on both
                sides). Test labels never touch a threshold.

  python3 scripts/57_eval_readout.py --mode probe --adapter results/adapters/readout_probe
  python3 scripts/57_eval_readout.py --mode probe --adapter none
"""
import argparse
import json
import os
import random

import mlx.core as mx
import numpy as np
import pandas as pd
from _bootstrap import load_cfg
from mlx.utils import tree_flatten
from sklearn.metrics import roc_auc_score

from mlx_vlm.utils import load
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import iterate_batches
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model

from src.eval_accuracy import RELIABLE_CLASSES, propagate_hierarchy
from src.grouping import CORE_PATHOLOGY_GROUPS, to_groups
from src.readout_ft import FINDABLE, INSTRUCTION, N_SLOTS, check_tokens, slot_probs, truth_set


def load_model(model_id, adapter, weights=None):
    model, processor = load(model_id, processor_config={"trust_remote_code": True})
    if adapter and adapter != "none":
        meta = json.load(open(os.path.join(adapter, "readout_meta.json")))
        model = get_peft_model(model, find_all_linear_names(model.language_model),
                               rank=meta["lora_rank"], alpha=meta["lora_alpha"],
                               dropout=0.0, verbose=False)
        w = mx.load(weights or os.path.join(adapter, "adapters.safetensors"))
        have = dict(tree_flatten(model.parameters()))
        missing = [k for k in w if k not in have or have[k].shape != w[k].shape]
        if missing:   # never silently score the base model as if it were the adapter
            raise RuntimeError(f"{len(missing)} adapter tensors have nowhere to land, "
                               f"e.g. {missing[:3]}")
        model.load_weights(list(w.items()), strict=False)
        k0 = next(iter(w))
        assert mx.allclose(dict(tree_flatten(model.parameters()))[k0], w[k0]).item()
        print(f"  adapter loaded: {len(w)} tensors verified in place", flush=True)
    model.eval()
    check_tokens(processor.tokenizer)
    return model, processor


def image_for(row):
    for col in ("frontal_paths", "lateral_paths"):   # lateral fallback = Exp 1 convention
        v = row[col]
        for p in (eval(v) if isinstance(v, str) else []):
            if p and os.path.exists(p):
                return p
    return None


def prompt_inputs(model, processor, image):
    """Exact training prefix: build the training example with a dummy answer and
    cut it where the completion mask starts, so prompt tokens match 56 byte for byte."""
    rec = [{"image": image, "messages": [
        {"role": "user", "content": INSTRUCTION},
        {"role": "assistant", "content": "\n".join(["no"] * N_SLOTS)}]}]
    ds = VisionDataset(rec, model.config.__dict__, processor, train_on_completions=True)
    b = next(iterate_batches(ds, 1, 2048, train=False))
    start = int(np.argmax(np.array(b["completion_mask"][0])))
    return b["input_ids"][:, :start], b["pixel_values"]


def run(model, processor, studies, uids, out_csv):
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    have = set(done.uid) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    for i, u in enumerate(uids):
        if u in have:
            continue
        img = image_for(studies.loc[u])
        if img is None:
            continue
        ids, pv = prompt_inputs(model, processor, img)
        p, top1, mass = slot_probs(model, ids, pv)
        rows.append({"uid": u, **{f"p_{c}": v for c, v in zip(FINDABLE, p)},
                     "fmt_top1": top1.mean(), "ans_mass": mass.mean()})
        if (i + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)   # resumable
            print(f"  {i + 1}/{len(uids)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df.set_index("uid")


def matrices(df, studies):
    uids = list(df.index)
    P = df[[f"p_{c}" for c in FINDABLE]].to_numpy()
    T = np.array([[1 if c in truth_set(studies.loc[u]) else 0 for c in FINDABLE] for u in uids])
    return uids, P, T


def micro(yt, yp):
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def ladder(pred_sets, T, label):
    pred = np.array([[1 if c in propagate_hierarchy(s) else 0 for c in FINDABLE]
                     for s in pred_sets])
    rel = [FINDABLE.index(c) for c in RELIABLE_CLASSES if c in FINDABLE]
    k = list(CORE_PATHOLOGY_GROUPS)
    def grp(M):
        return np.array([[1 if g in to_groups({FINDABLE[j] for j in np.where(r)[0]},
                                             CORE_PATHOLOGY_GROUPS) else 0 for g in k] for r in M])
    f13 = micro(T, pred); frel = micro(T[:, rel], pred[:, rel]); f3 = micro(grp(T), grp(pred))
    bacc = float(((T.sum(1) > 0) == (pred.sum(1) > 0)).mean())
    print(f"  {label:<34} 13cls {f13[2]:.4f} (P {f13[0]:.3f} R {f13[1]:.3f})  "
          f"rel11 {frel[2]:.4f}  3grp {f3[2]:.4f}  binAcc {bacc:.4f}  "
          f"find/std {pred.sum(1).mean():.2f}")
    return f13[2]


def probe_report(df, studies, label):
    uids, P, T = matrices(df, studies)
    print(f"\n=== PROBE: {label}  (n={len(uids)}, {T.any(1).mean():.0%} abnormal) ===")
    print(f"  format: top-1 token is yes/no on {df.fmt_top1.mean():.1%} of slots; "
          f"mean P(yes or no) = {df.ans_mass.mean():.3f}")
    yes = (P >= 0.5)
    print(f"  greedy yes-rate {yes.mean():.1%} of cells (truth {T.mean():.1%}); "
          f"distinct yes/no patterns {len({tuple(r) for r in yes})} of {len(uids)}")
    print(f"  {'finding':<28}{'pos':>4} {'meanP':>7} {'sdP':>7} {'yes%':>6} {'AUROC':>7}")
    aucs = []
    for j, c in enumerate(FINDABLE):
        a = roc_auc_score(T[:, j], P[:, j]) if 0 < T[:, j].sum() < len(T) else np.nan
        if T[:, j].sum() >= 5:
            aucs.append(a)
        print(f"  {c:<28}{T[:, j].sum():>4} {P[:, j].mean():>7.3f} {P[:, j].std():>7.3f} "
              f"{yes[:, j].mean():>6.0%} {a:>7.3f}")
    pooled = roc_auc_score(T.ravel(), P.ravel())
    print(f"  pooled-cell AUROC {pooled:.3f} | macro AUROC (classes with >=5 pos) "
          f"{np.nanmean(aucs):.3f} over {len(aucs)} classes")
    sd = P.std(0).mean()
    print(f"  VERDICT inputs: mean across-study sd of P(yes) = {sd:.3f} "
          f"({'VARIES' if sd > 0.02 else 'FLAT - collapse signature'})")


def full_eval(dev_df, test_df, studies):
    du, dP, dT = matrices(dev_df, studies)
    tu, tP, tT = matrices(test_df, studies)
    thr = np.full(N_SLOTS, 0.5)
    for j in range(N_SLOTS):
        cands = np.unique(np.quantile(dP[:, j], np.linspace(0, 1, 201)))
        f = [micro(dT[:, j], (dP[:, j] >= t).astype(int))[2] for t in cands]
        if dT[:, j].sum() > 0:
            thr[j] = cands[int(np.argmax(f))]
    print("\n=== FULL: per-class thresholds chosen on DEV, applied once to TEST ===")
    for j, c in enumerate(FINDABLE):
        print(f"  {c:<28} thr {thr[j]:.3f}  test calls {(tP[:, j] >= thr[j]).sum():>4} "
              f"true {tT[:, j].sum():>4}  AUROC "
              f"{roc_auc_score(tT[:, j], tP[:, j]) if 0 < tT[:, j].sum() < len(tT) else float('nan'):.3f}")
    sets = lambda P, t: [{FINDABLE[j] for j in np.where(r >= t)[0]} for r in P]
    ladder(sets(tP, np.full(N_SLOTS, 0.5)), tT, "TEST, fixed 0.5 threshold")
    ladder(sets(tP, thr), tT, "TEST, dev-tuned per-class threshold")
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/medgemma-4b-it-8bit")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the base model")
    ap.add_argument("--mode", choices=["probe", "full"], default="probe")
    ap.add_argument("--n", type=int, default=80, help="probe subset size (dev)")
    ap.add_argument("--weights", default=None,
                    help="score a specific checkpoint file (e.g. 0000300_adapters.safetensors)")
    ap.add_argument("--set", choices=["probe", "confirm"], default="probe",
                    help="probe = the 80-study SELECTION set; confirm = a DISJOINT dev "
                         "set used once to accept/reject the selected checkpoint")
    ap.add_argument("--n-confirm", type=int, default=120)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = load_cfg("configs/config_medgemma_estr.yaml")
    out = cfg["paths"]["out_dir"]
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    dev = [u for u in pd.read_csv(os.path.join(out, "dev_manifest.csv")).uid if u in studies.index]
    test = [u for u in pd.read_csv(os.path.join(out, "test_manifest.csv")).uid if u in studies.index]
    tag = args.tag or ("base" if args.adapter == "none" else os.path.basename(args.adapter.rstrip("/")))
    model, processor = load_model(args.model, args.adapter, args.weights)

    if args.mode == "probe":
        # selection set: unchanged sampling, so it is the same 80 studies as before
        rng = random.Random(cfg.get("seed", 20260705))
        abn = [u for u in dev if truth_set(studies.loc[u])]
        nrm = [u for u in dev if not truth_set(studies.loc[u])]
        uids = rng.sample(abn, args.n // 2) + rng.sample(nrm, args.n - args.n // 2)
        if args.set == "confirm":
            sel = set(uids)
            rng2 = random.Random(cfg.get("seed", 20260705) + 1)
            abn2 = [u for u in abn if u not in sel]
            nrm2 = [u for u in nrm if u not in sel]
            uids = (rng2.sample(abn2, args.n_confirm // 2)
                    + rng2.sample(nrm2, args.n_confirm - args.n_confirm // 2))
            assert not sel & set(uids)
        df = run(model, processor, studies, uids,
                 os.path.join(out, f"readout_slots_{tag}_{args.set}.csv"))
        probe_report(df, studies, f"{tag} [{args.set}]")
    else:
        d = run(model, processor, studies, dev, os.path.join(out, f"readout_slots_{tag}_dev.csv"))
        t = run(model, processor, studies, test, os.path.join(out, f"readout_slots_{tag}_test.csv"))
        probe_report(t, studies, f"{tag} (test)")
        full_eval(d, t, studies)


if __name__ == "__main__":
    main()
