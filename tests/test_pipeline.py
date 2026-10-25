"""Fast unit tests for the deterministic core. Run: python -m pytest tests/ -q
(or: python tests/test_pipeline.py for a no-pytest smoke run)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.labeling import label_report, is_hedged
from src.normalize import normalize_dx, primary_class
from src.eval_calibration import calibration_metrics
from src.eval_groundedness import extract_claims, adjudicate_claim
from src.schema import validate, coerce, extract_json
from src import stats


def test_labeler_positive_and_negation():
    lab = label_report("There is a large pleural effusion. No pneumothorax.")
    assert lab["Pleural Effusion"] == 1.0
    assert lab["Pneumothorax"] == 0.0


def test_labeler_uncertainty():
    lab = label_report("Findings may represent pneumonia.")
    assert lab["Pneumonia"] == -1.0
    assert is_hedged("cannot exclude consolidation")


def test_labeler_normal():
    lab = label_report("Normal chest. Lungs are clear. No acute disease.")
    assert lab["No Finding"] == 1.0


def test_borderline_not_device():
    # regression: 'borderline' must NOT trigger Support Devices 'line'
    lab = label_report("Borderline cardiomegaly.")
    assert np.isnan(lab["Support Devices"])


def test_normalize():
    assert "Cardiomegaly" in normalize_dx("enlarged cardiac silhouette")
    assert primary_class("no acute cardiopulmonary finding") == "No Finding"


def test_calibration_perfect():
    conf = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([1, 1, 0, 0])
    m = calibration_metrics(conf, correct, n_bins=10)
    assert m["ece"] < 1e-9 and m["brier"] < 1e-9


def test_calibration_overconfident():
    conf = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([0, 0, 0, 0])
    m = calibration_metrics(conf, correct, n_bins=10)
    assert m["overconfidence_rate"] == 1.0 and m["ece"] > 0.5


def test_groundedness_adjudication():
    ref = pd.Series({"lbl_Pleural Effusion": 1.0, "lbl_Pneumothorax": 0.0})
    supp = adjudicate_claim({"cls": "Pleural Effusion", "presence": "present"}, ref)
    contra = adjudicate_claim({"cls": "Pneumothorax", "presence": "present"}, ref)
    assert supp == "supported" and contra == "contradicted"


def test_schema():
    assert validate({"primary_diagnosis": "x", "confidence": 80, "explanation": "y"}) == []
    fixed = coerce({"primary_diagnosis": "x", "confidence": 130, "explanation": "y"})
    assert fixed["confidence"] == 100
    assert extract_json('```json\n{"a":1}\n```') == {"a": 1}


def test_mcnemar_and_wilcoxon_run():
    df = pd.DataFrame({
        "uid": [1, 1, 2, 2, 3, 3],
        "prompt_condition": ["a", "b"] * 3,
        "correct": [1, 0, 1, 1, 0, 1],
    })
    mc = stats.mcnemar_all_pairs(df, "correct")
    assert len(mc) == 1 and "p_value" in mc.columns


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
