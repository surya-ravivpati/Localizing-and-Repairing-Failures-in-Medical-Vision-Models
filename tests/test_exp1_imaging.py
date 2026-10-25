"""Experiment 1 — locks the properties that make arm A a valid control.

The whole ablation rests on 'baseline reproduces the current pipeline exactly'.
These tests fail loudly if a future edit breaks that.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import imaging                      # noqa: E402
from src.inference import _load_image_bytes  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


def _a_study():
    p = os.path.join(RESULTS, "studies_labeled.csv")
    if not os.path.exists(p):
        pytest.skip("studies_labeled.csv not built")
    df = pd.read_csv(p, nrows=40)
    df["frontal_paths"] = df["frontal_paths"].map(
        lambda s: eval(s) if isinstance(s, str) else s)
    df["lateral_paths"] = df["lateral_paths"].map(
        lambda s: eval(s) if isinstance(s, str) else s)
    for _, r in df.iterrows():
        if imaging._head_existing(r["frontal_paths"]):
            return r
    pytest.skip("no readable frontal image on this machine")


def test_head_existing_mirrors_backend_slicing():
    """GeminiBackend takes paths[:1] and drops it if missing -- NOT 'first that
    exists'. Arm A must select identically or it is not a reproduction."""
    assert imaging._head_existing([]) == []
    assert imaging._head_existing(None) == []
    assert imaging._head_existing(["/nope/missing.png"]) == []
    # a missing FIRST path must not fall through to a present second path
    assert imaging._head_existing(["/nope/missing.png", __file__]) == []
    assert imaging._head_existing([__file__]) == [__file__]


def test_baseline_render_is_byte_identical_to_live_pipeline(tmp_path):
    """Arm A bytes == what the live backend would have sent at image_max_side=1024."""
    row = _a_study()
    parts = imaging.render_variant(row, "baseline", str(tmp_path))
    assert parts, "expected at least a frontal part"
    for part in parts:
        expected, _ = _load_image_bytes(part["src_path"], imaging.BASELINE_MAX_SIDE)
        assert open(part["path"], "rb").read() == expected, (
            f"{part['kind']} render diverged from the live encoder")


def test_baseline_downscales_to_1024_and_keeps_aspect():
    row = _a_study()
    src = imaging._head_existing(row["frontal_paths"])[0]
    from PIL import Image
    with Image.open(src) as im:
        sw, sh = im.size
    data, _ = _load_image_bytes(src, imaging.BASELINE_MAX_SIDE)
    w, h = imaging._dims(data)
    assert max(w, h) <= imaging.BASELINE_MAX_SIDE
    assert abs((w / h) - (sw / sh)) < 0.02, "aspect ratio drifted"


def test_manifest_columns_are_stable():
    """35_run_twostage and 38_eval read these by name across every arm."""
    for c in ("uid", "variant", "part_idx", "kind", "zone", "path",
              "width", "height", "sha256", "src_path"):
        assert c in imaging.MANIFEST_COLUMNS


def test_highres_delivers_more_pixels_than_baseline(tmp_path):
    """Arm B must actually increase pixel count, and must never UPSCALE."""
    row = _a_study()
    base = imaging.render_variant(row, "baseline", str(tmp_path / "b"))
    hi = imaging.render_variant(row, "highres", str(tmp_path / "h"))
    assert len(base) == len(hi), "arms must send the same number of parts"
    from PIL import Image
    for b, h in zip(base, hi):
        with Image.open(b["src_path"]) as im:
            native = max(im.size)
        assert max(h["width"], h["height"]) <= min(imaging.HIGHRES_MAX_SIDE, native), \
            "highres must never upscale beyond the native image"
        assert max(h["width"], h["height"]) > max(b["width"], b["height"]), \
            "highres must exceed baseline resolution"


def test_patient_right_maps_to_image_left():
    """RADIOGRAPHIC CONVENTION: patient's right = IMAGE's left in a frontal view.
    Getting this backwards mirrors every crop and silently invalidates arm C/D."""
    W = H = 300
    l0, t0, r0, b0 = imaging.zone_box(("upper", "right"), W, H)   # patient RIGHT
    l1, t1, r1, b1 = imaging.zone_box(("upper", "left"), W, H)    # patient LEFT
    assert l0 == 0, "patient-right crop must start at the image's left edge"
    assert r1 == W, "patient-left crop must end at the image's right edge"
    assert (l0 + r0) / 2 < (l1 + r1) / 2, "patient-right must sit left of patient-left"


def test_zone_bands_are_ordered_and_overlap():
    W = H = 300
    tops = [imaging.zone_box((b, "right"), W, H)[1] for b in imaging.ZONE_BANDS]
    assert tops == sorted(tops), "upper/mid/lower must run top to bottom"
    up = imaging.zone_box(("upper", "right"), W, H)
    mid = imaging.zone_box(("mid", "right"), W, H)
    assert up[3] > mid[1], "adjacent bands must overlap (margin) so findings survive"
    for z in imaging.ZONES:                      # never escape the frame
        l, t, r, b = imaging.zone_box(z, W, H)
        assert 0 <= l < r <= W and 0 <= t < b <= H


def test_multicrop_sends_full_view_plus_six_zones(tmp_path):
    row = _a_study()
    parts = imaging.render_variant(row, "multicrop_f", str(tmp_path))
    assert len(parts) == 7, f"expected full + 6 zones, got {len(parts)}"
    assert parts[0]["kind"] in ("frontal", "lateral") and parts[0]["zone"] == ""
    assert [p["zone"] for p in parts[1:]] == [f"{s}_{b}" for b in imaging.ZONE_BANDS
                                              for s in imaging.ZONE_SIDES]


def test_mesh_zone_parsing_and_laterality():
    mz = imaging.mesh_zone
    assert mz("Consolidation/lung/base/left") == ("lower", "left")
    assert mz("Opacity/lung/apex/right/irregular") == ("upper", "right")
    assert mz("Airspace Disease/lung/lower lobe/left/severe") == ("lower", "left")
    assert mz("Density/cardiophrenic angle/left") == ("mid", "left")
    # not localizable: global, bilateral, empty
    assert mz("Cardiomegaly/borderline") is None
    assert mz("Opacity/lung/bilateral/interstitial/diffuse") is None
    assert mz("") is None and mz(None) is None
    # first entry carrying BOTH band and side wins (deterministic)
    assert mz("Cardiomegaly/mild;Consolidation/lung/apex/right") == ("upper", "right")


def test_oracle_assigns_exactly_one_zone_to_every_study():
    """Anti-leak: crop presence must not signal abnormality, so normals and
    non-localizable studies get a zone too, drawn from the genuine marginal."""
    pairs = [(1, "Consolidation/lung/base/left"), (2, "Opacity/lung/apex/right"),
             (3, "Cardiomegaly/borderline"), (4, ""), (5, None)]
    a = imaging.assign_oracle_zones(pairs, seed=7)
    assert set(a) == {1, 2, 3, 4, 5}, "every study must receive a zone"
    assert a[1] == (("lower", "left"), "localizable")
    assert a[2] == (("upper", "right"), "localizable")
    for uid in (3, 4, 5):
        assert a[uid][1] == "assigned"
        assert a[uid][0] in imaging.ZONES
    assert imaging.assign_oracle_zones(pairs, seed=7) == a, "must be reproducible"


def test_wrong_zone_is_never_the_true_zone():
    """The whole control depends on the crop being genuinely wrong."""
    pairs = [(1, "Consolidation/lung/base/left"), (2, "Opacity/lung/apex/right"),
             (3, "Cardiomegaly/borderline")]
    w = imaging.assign_wrong_zones(pairs, seed=11)
    assert set(w) == {1, 2}, "studies with no true zone must be excluded"
    assert w[1][0] != ("lower", "left") and w[1][0] in imaging.ZONES
    assert w[2][0] != ("upper", "right") and w[2][0] in imaging.ZONES
    assert all(v[1] == "wrong" for v in w.values())
    assert imaging.assign_wrong_zones(pairs, seed=11) == w, "must be reproducible"
