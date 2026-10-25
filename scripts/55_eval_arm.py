"""Three-arm alignment intervention: evaluation side.

Scores an arm on the locked 754-study test split and, more importantly, on the
metric the intervention is actually for:

  LATENT-TO-EXPRESSED CONVERSION.  Section 9.6 found 129 findings that the model
  itself failed to report but that a linear probe recovered from the very vectors
  the model was handed. Those 129 are exactly what an alignment intervention has
  to convert from recoverable into said. Baseline scores 0/129 by construction,
  so the metric has a true zero and needs no separate control.

  conversion = |{cells in the 129 that this arm now expresses}| / 129

That is a far sharper read than overall F1, which can move because the arm calls
more of everything. Both are reported, plus the flag rate, so a conversion bought
by indiscriminate over-calling is visible rather than hidden.

Prompt parity is enforced, not assumed: the arm is generated with the same
`perception_only` scaffold as the baseline Stage-A run, and the script refuses to
proceed if the baseline file used a different condition.

ALWAYS READ THE RAW GENERATIONS. A broken adapter and a collapsed one produce the
same F1 once normalize_dx maps their output to the empty set; three runs were lost
to that before a no-adapter control caught it. The output-diversity block at the
end is not decoration.

Usage
  python3 scripts/55_eval_arm.py --arm-dir results/arms/connector --limit 40   # probe
  python3 scripts/55_eval_arm.py --arm-dir results/arms/connector              # full
  python3 scripts/55_eval_arm.py --baseline                                    # sanity
"""
import argparse
import ast
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from _bootstrap import load_cfg
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import prompts
from src.eval_accuracy import (_pred_set, _truth_positive_set, propagate_hierarchy,
                               RELIABLE_CLASSES)
from src.grouping import CORE_PATHOLOGY_GROUPS, to_groups, any_acute
from src.labeling import CHEXPERT_CLASSES
from src.normalize import normalize_dx

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ALL = [c for c in CHEXPERT_CLASSES if c != "No Finding"]
C_GRID = [0.001, 0.01, 0.1, 1.0]


def truth_matrix(studies, uids, classes, prefix="lblcx_"):
    by = studies.set_index("uid")
    return np.array([[1 if c in propagate_hierarchy(
        _truth_positive_set(by.loc[u], "ones", prefix)) else 0 for c in classes]
        for u in uids], dtype=int)


def propagate_matrix(M, classes):
    return np.array([[1 if c in propagate_hierarchy(
        {k for k, v in zip(classes, r) if v}) else 0 for c in classes] for r in M], dtype=int)


def micro(yt, yp):
    tp = int((yt & yp).sum()); fp = int(((1 - yt) & yp).sum()); fn = int((yt & (1 - yp)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ladder(pred, truth):
    def sub(M, cls):
        idx = [ALL.index(c) for c in cls]
        return M[:, idx]
    out = {}
    out["f1_13"] = micro(truth, pred)[2]
    out["f1_rel11"] = micro(sub(truth, RELIABLE_CLASSES), sub(pred, RELIABLE_CLASSES))[2]
    g = sorted(CORE_PATHOLOGY_GROUPS)

    def gm(M):
        sets = [{c for c, v in zip(ALL, r) if v} for r in M]
        return np.array([[1 if x in to_groups(s, CORE_PATHOLOGY_GROUPS) else 0
                          for x in g] for s in sets], dtype=int)
    out["f1_3grp"] = micro(gm(truth), gm(pred))[2]
    bp = np.array([any_acute({c for c, v in zip(ALL, r) if v}) for r in pred], dtype=int)
    bt = np.array([any_acute({c for c, v in zip(ALL, r) if v}) for r in truth], dtype=int)
    out["bin_acc"] = float((bp == bt).mean())
    out["flag_rate"] = float(bp.mean())
    return out


def interface_recovered(out_dir, studies, te_u, ute, vlm_raw, vlm, yte, seed):
    """The 129: cells the model missed that a matched-commitment interface probe
    recovers. Cached, because refitting costs minutes and never changes."""
    cache = os.path.join(out_dir, "exp6_interface_recovered.npy")
    if os.path.exists(cache):
        m = np.load(cache)
        if m.shape == yte.shape:
            return m
    fd = os.path.join(out_dir, "exp6_features", "8bit")
    de_u = list(pd.read_csv(os.path.join(out_dir, "dev_manifest.csv")).uid)
    tr_u = [u for u in studies.uid if u not in set(te_u) | set(de_u)]

    def load(uids):
        have = [u for u in uids if os.path.exists(os.path.join(fd, f"{u}.npz"))]
        return np.stack([np.load(os.path.join(fd, f"{u}.npz"))["proj"] for u in have]), have
    Xtr, utr = load(tr_u); Xde, ude = load(de_u); Xte, _ = load(ute)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xde, Xte = sc.transform(Xtr), sc.transform(Xde), sc.transform(Xte)
    ytr, yde = truth_matrix(studies, utr, ALL), truth_matrix(studies, ude, ALL)
    scores = np.zeros((len(ute), len(ALL)))
    for ci in range(len(ALL)):
        if ytr[:, ci].sum() < 2 or len(np.unique(yde[:, ci])) < 2:
            scores[:, ci] = -1e9; continue
        best, ba = None, -1
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                     random_state=seed).fit(Xtr, ytr[:, ci])
            a = roc_auc_score(yde[:, ci], clf.decision_function(Xde))
            if a > ba:
                best, ba = clf, a
        scores[:, ci] = best.decision_function(Xte)
    yp = np.zeros_like(vlm_raw)
    for ci in range(len(ALL)):
        k = int(vlm_raw[:, ci].sum())
        if k:
            yp[np.argsort(-scores[:, ci])[:k], ci] = 1
    rec = ((yte == 1) & (vlm == 0) & (propagate_matrix(yp, ALL) == 1))
    np.save(cache, rec)
    return rec


def generate(arm_dir, model_id, uids, studies, condition, limit, ckpt=None):
    """Resumable: every generation is appended to `ckpt` as it is produced, and a
    restart skips what is already there. A 754-study pass takes hours on this
    hardware and a Metal GPU timeout at study ~550 previously discarded the lot,
    because rows were only held in memory."""
    from mlx_vlm import load as vload, generate as vgen
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx.utils import tree_unflatten, tree_flatten
    import mlx.core as mx

    model, processor = vload(model_id, processor_config={"trust_remote_code": True})
    if arm_dir:
        wf = os.path.join(arm_dir, "adapters.safetensors")
        w = mx.load(wf)
        # Arm `connector` saves plain module weights, which update() can apply to a
        # freshly loaded model. Arm `both` also saves lora_a/lora_b, and those
        # parameters have nowhere to land unless the LoRA layers are re-created
        # first -- update() would otherwise drop them silently and evaluate the
        # untouched base model while reporting success.
        lora_keys = [k for k in w if k.endswith((".lora_a", ".lora_b"))]
        if lora_keys:
            from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
            rank = int(w[lora_keys[0]].shape[1] if w[lora_keys[0]].ndim > 1 else 8)
            model = get_peft_model(model, find_all_linear_names(model.language_model),
                                   rank=rank, alpha=2 * rank, dropout=0.0, verbose=False)
            print(f"  re-created LoRA layers (rank {rank}) for {len(lora_keys)} adapter tensors")
        before = {k: v for k, v in tree_flatten(model.parameters())
                  if k in w}
        model.update(tree_unflatten(list(w.items())))
        mx.eval(model.parameters())
        after = dict(tree_flatten(model.parameters()))
        applied = sum(1 for k in w if k in after
                      and not mx.array_equal(after[k], before.get(k, after[k] * 0 + 1e9)))
        missing = [k for k in w if k not in after]
        print(f"  loaded {len(w)} trained tensors from {wf}")
        if missing:
            raise SystemExit(f"  {len(missing)} tensors had nowhere to land, e.g. "
                             f"{missing[:3]} — the eval would silently score the base model")
        print(f"  verified all {len(w)} landed on real parameters")
    done = {}
    if ckpt and os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        done = dict(zip(prev.uid, prev.raw.astype(str)))
        print(f"  resuming: {len(done)} already generated")
    todo = uids[:limit] if limit else uids
    rows = [{"uid": u, "raw": done[u]} for u in todo if u in done]
    for i, uid in enumerate(todo):
        if uid in done:
            continue
        r = studies.loc[uid]
        raw = r.get("frontal_paths")
        paths = ast.literal_eval(raw) if isinstance(raw, str) and raw.strip() else []
        paths = [p for p in paths if p and os.path.exists(p)][:1]
        system, user = prompts.build_freetext(condition, r.get("indication", ""))
        prompt = apply_chat_template(
            processor, model.config,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            num_images=len(paths))
        o = vgen(model, processor, prompt, image=paths or None, max_tokens=256,
                 temperature=0.0, verbose=False)
        text = (o.text if hasattr(o, "text") else str(o)).strip()
        rows.append({"uid": uid, "raw": text})
        if ckpt:
            pd.DataFrame(rows).to_csv(ckpt, index=False)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(todo)}", flush=True)
    order = {u: k for k, u in enumerate(todo)}
    return pd.DataFrame(rows).sort_values("uid", key=lambda c: c.map(order)).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", default=None, help="omit (or --baseline) for untrained")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--model", default="mlx-community/medgemma-4b-it-8bit")
    ap.add_argument("--stage-a", default="predictions_test_twostage_mg_stageA.csv")
    ap.add_argument("--condition", default="perception_only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out-suffix", default=None)
    args = ap.parse_args()

    out = load_cfg()["paths"]["out_dir"]
    studies = pd.read_csv(os.path.join(out, "studies_labeled.csv")).set_index("uid")
    te_u = list(pd.read_csv(os.path.join(out, "test_manifest.csv")).uid)
    base = pd.read_csv(os.path.join(out, args.stage_a))
    got = set(base.prompt_condition.unique())
    if got != {args.condition}:
        raise SystemExit(f"prompt parity violated: baseline ran {got}, "
                         f"this eval would run {args.condition!r}")
    base = base.set_index("uid")

    arm_dir = None if args.baseline else args.arm_dir
    tag = args.out_suffix or ("baseline" if arm_dir is None
                              else os.path.basename(arm_dir.rstrip("/")))
    print(f"[eval:{tag}] condition={args.condition} "
          f"{'(untrained baseline)' if arm_dir is None else arm_dir}")

    ckpt = os.path.join(out, f"arm_predictions_{tag}.partial.csv") if not args.limit else None
    gen = generate(arm_dir, args.model, te_u, studies, args.condition, args.limit, ckpt)
    path = os.path.join(out, f"arm_predictions_{tag}.csv")
    gen.to_csv(path, index=False)
    ute = list(gen.uid)

    pred = propagate_matrix(np.array(
        [[1 if c in {x for x in normalize_dx(t, negation=True) if x != "No Finding"}
          else 0 for c in ALL] for t in gen.raw]), ALL)
    yte = truth_matrix(studies.reset_index(), ute, ALL)

    vlm_raw = np.array([[1 if c in _pred_set(base.loc[u]) else 0 for c in ALL]
                        if u in base.index else [0] * len(ALL) for u in ute])
    vlm = propagate_matrix(vlm_raw, ALL)

    lb, la = ladder(vlm, yte), ladder(pred, yte)
    print(f"\n=== ladder, n={len(ute)} (CheXbert ref, hierarchy-aware) ===")
    print(f"  {'':<22}{'13cls':>8}{'rel11':>8}{'3grp':>8}{'binAcc':>8}{'flag%':>8}")
    for nm, d in (("baseline Stage A", lb), (f"arm: {tag}", la)):
        print(f"  {nm:<22}{d['f1_13']:>8.4f}{d['f1_rel11']:>8.4f}"
              f"{d['f1_3grp']:>8.4f}{d['bin_acc']:>8.4f}{100*d['flag_rate']:>7.1f}%")

    if not args.limit:
        rec = interface_recovered(out, studies.reset_index(), te_u, ute,
                                  vlm_raw, vlm, yte, args.seed)
        n129 = int(rec.sum())
        conv = int((rec & (pred == 1)).sum())
        print(f"\n=== latent-to-expressed conversion ===")
        print(f"  the probe-recovered set: {n129} cells (baseline expresses 0 by construction)")
        print(f"  this arm expresses:      {conv}/{n129} = {100*conv/max(n129,1):.1f}%")
        miss = (yte == 1) & (vlm == 0)
        print(f"  (all model misses: {int(miss.sum())}; this arm expresses "
              f"{int((miss & (pred == 1)).sum())} of them)")

    # hash the WHOLE generation, not a prefix: a fine-tuned model often shares an
    # opening preamble across every study while the findings that follow differ,
    # and a prefix-keyed counter reports that as near-total collapse.
    c = Counter(str(t) for t in gen.raw)
    lens = gen.raw.astype(str).str.len()
    print(f"\n=== RAW OUTPUT DIVERSITY: {len(c)} distinct of {len(gen)} "
          f"| chars mean {lens.mean():.0f} min {lens.min()} max {lens.max()} ===")
    for k, v in c.most_common(5):
        print(f"  {v:>4}  {k[:70]!r}")
    if len(c) <= 3:
        print("  WARNING: near-degenerate output. Check the raw text before "
              "reading any metric above as a result.")
    print(f"\n[eval:{tag}] wrote {path}")


if __name__ == "__main__":
    main()
