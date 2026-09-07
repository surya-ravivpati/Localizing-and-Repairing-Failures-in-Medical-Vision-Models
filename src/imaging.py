"""Pre-rendered image variants for Experiment 1 (visual-information ablation).

Experiment 1 varies ONLY the visual input to the two-stage pipeline. To keep that
isolation airtight, every arm renders its image parts to disk HERE, and inference
then runs with `image_max_side: 0` (raw-bytes passthrough). Three confounds die
at once:

  1. `inference._load_image_bytes` takes DIFFERENT code paths for max_side=0 (raw
     file bytes) vs max_side=N (PIL resize -> convert("L") -> PNG re-encode).
     Pre-rendering puts every arm on the identical path.
  2. All resize/crop decisions live in one inspectable module instead of being
     implicit in per-arm backend config.
  3. Every part becomes a file you can open and look at BEFORE spending API budget.

Arm A (`baseline`) is defined to reproduce the CURRENT pipeline exactly: it calls
`_load_image_bytes(path, 1024)` — the very function the live backend uses — and
writes those bytes verbatim, so byte-identity is true by construction and locked
by a unit test rather than re-implemented and hoped for.

Only the variants required by the currently-authorized condition are implemented;
later arms extend IMPLEMENTED_VARIANTS as they are built.
"""
from __future__ import annotations

import hashlib
import io
import os

from .inference import _load_image_bytes

# Arm A reproduces the live default (`image_max_side: 1024`).
BASELINE_MAX_SIDE = 1024
# Arm B: 2x the linear resolution = 4x the pixels. Native images are ~2494x2048
# (p5 2048), so 2048 is a real increase that never UPSCALES anything -- PIL only
# resizes when max(size) > max_side, so images already at/below 2048 pass through
# at native size. Same encode path as arm A; only the pixel count differs.
HIGHRES_MAX_SIDE = 2048

# MEASURED Gemini behaviour (diagnostic, 2026-08-16), prompt_token_count for one study:
#     media_res=HIGH, 1 image : 1024px -> 1643 | 2048px -> 2933   (scales 1.79x)
#     media_res=HIGH, 2 images: 1024px ->  869 | 2048px ->  869   (FLAT)
#     media_res=unset, 1 image: 1024px ->  611 | 2048px ->  611   (FLAT)
# A second image part makes the API ignore MEDIA_RESOLUTION_HIGH and pin the payload,
# so resolution can only be manipulated with exactly ONE image. The `_f` (frontal-only)
# variants are therefore the valid ablation series; the frontal+lateral pair is kept
# because arm A in that configuration is what ties this harness to the historical
# pipeline (paired McNemar p=1.000).
VARIANT_SPECS = {
    "baseline":    {"max_side": BASELINE_MAX_SIDE, "lateral": True,  "crops": None},
    "highres":     {"max_side": HIGHRES_MAX_SIDE,  "lateral": True,  "crops": None},
    "baseline_f":  {"max_side": BASELINE_MAX_SIDE, "lateral": False, "crops": None},
    "highres_f":   {"max_side": HIGHRES_MAX_SIDE,  "lateral": False, "crops": None},
    # Arm C: the full frontal plus each of the 6 standard clinical zones, every crop
    # cut from the NATIVE image then resized to 1024 -- so each region arrives at far
    # higher effective resolution than it has in the whole-image view.
    "multicrop_f": {"max_side": BASELINE_MAX_SIDE, "lateral": False, "crops": "all6"},
    # Arm D: the full frontal plus ONE crop -- the zone where the finding actually is.
    # Leaky by design; an upper bound on what perfect spatial guidance could buy.
    "oracle_f":    {"max_side": BASELINE_MAX_SIDE, "lateral": False, "crops": "oracle"},
    # EXPERIMENT 4 — wrong-crop control. Identical rendering to oracle_f; only the
    # zone differs. Run on the studies that HAVE a true zone, so the comparison is
    # correct-vs-wrong crop on the same cases at the same payload, with none of the
    # case-mix asymmetry that forced the difference-in-differences in Exp 1.
    "wrongcrop_f": {"max_side": BASELINE_MAX_SIDE, "lateral": False, "crops": "oracle"},
}

# EXPERIMENT 2 — resolution dose-response. Single image throughout (the only
# configuration in which resolution reaches the model at all). 1024 and 2048 are
# already covered by baseline_f / highres_f, so only the missing rungs are defined
# here; the sweep is scored against those existing runs on the same subset.
for _px in (256, 512, 768, 1536):
    VARIANT_SPECS[f"res{_px}_f"] = {"max_side": _px, "lateral": False, "crops": None}

# EXPERIMENT 3 — controlled visual degradation. Experiment 2 showed a quarter-
# resolution image reads as well as the baseline, which raises a sharper question
# than "how much detail helps": is the image being used at all? A degradation
# ladder needs a FLOOR to answer that, so `blank` (a uniform grey field carrying
# zero information) is included alongside the graded degradations. Any score it
# attains is what the model gets from priors — the clinical indication, the base
# rate, and the prompt — with no image evidence whatsoever. Every rung stays at
# 1024px and one image part, so resolution and part-count are held constant and
# only image CONTENT is degraded.
for _name, _spec in (("blur4",    {"degrade": ("blur", 4)}),
                     ("blur12",   {"degrade": ("blur", 12)}),
                     ("scramble", {"degrade": ("scramble", 8)}),
                     ("blank",    {"degrade": ("blank", 0)})):
    VARIANT_SPECS[f"{_name}_f"] = {"max_side": BASELINE_MAX_SIDE, "lateral": False,
                                   "crops": None, **_spec}

# The standard chest-radiograph review zones, named by PATIENT side.
# RADIOGRAPHIC CONVENTION: the patient's right appears on the IMAGE's LEFT in a
# frontal view (confirmed on these images -- the "L" marker sits on the image right).
# Getting this backwards would mirror every crop, so it is asserted in a unit test.
ZONE_BANDS = ("upper", "mid", "lower")
ZONE_SIDES = ("right", "left")            # patient side
ZONES = [(b, sd) for b in ZONE_BANDS for sd in ZONE_SIDES]
ZONE_MARGIN = 0.15                        # overlap so findings on a boundary survive


def zone_box(zone, W: int, H: int, margin: float = ZONE_MARGIN) -> tuple:
    """Pixel box (left, top, right, bottom) for a (band, patient_side) zone.

    3 vertical bands x 2 halves, each expanded by `margin` and clipped. The images
    are NOT lung-field registered, so the overlap is deliberately generous."""
    band, side = zone
    i = ZONE_BANDS.index(band)
    top, bot = i / 3.0, (i + 1) / 3.0
    # patient right -> image LEFT half; patient left -> image RIGHT half
    x0, x1 = (0.0, 0.5) if side == "right" else (0.5, 1.0)
    dw, dh = (x1 - x0) * margin, (bot - top) * margin
    x0, x1 = max(0.0, x0 - dw), min(1.0, x1 + dw)
    top, bot = max(0.0, top - dh), min(1.0, bot + dh)
    return (int(x0 * W), int(top * H), int(x1 * W), int(bot * H))


def zone_caption(zone) -> str:
    band, side = zone
    return f"^ {side.capitalize()} {band} zone (patient {side}), higher resolution."


_BAND_TOKENS = {
    "upper": ("apex", "apical", "upper lobe"),
    "mid":   ("hilum", "hilar", "perihilar", "middle lobe", "lingula", "cardiophrenic"),
    "lower": ("base", "basilar", "lower lobe", "costophrenic", "diaphragm"),
}


def mesh_zone(mesh):
    """Parse the IU X-Ray `mesh` slash-hierarchy into a (band, patient_side) zone.

    e.g. "Consolidation/lung/base/left;Opacity/lung/apex/right" -> ("lower", "left").
    Returns the FIRST entry carrying BOTH a band and an explicit laterality, so the
    rule is deterministic on multi-finding studies. `bilateral` / `diffuse` and global
    findings (cardiomegaly, edema, support devices) carry no band+side pair and so
    come back None -- correctly, since there is no single region to crop."""
    if not isinstance(mesh, str) or not mesh.strip():
        return None
    for entry in mesh.split(";"):
        toks = [t.strip().lower() for t in entry.split("/") if t.strip()]
        band = side = None
        for t in toks:
            if side is None and t in ("left", "right"):
                side = t
            if band is None:
                for b, keys in _BAND_TOKENS.items():
                    if any(t == k or t.startswith(k) for k in keys):
                        band = b
                        break
        if band and side:
            return (band, side)
    return None


def assign_oracle_zones(uid_mesh_pairs, seed: int) -> dict:
    """-> {uid: (zone, "localizable"|"assigned")}, one zone for EVERY study.

    ANTI-LEAKAGE CONTROL. If only localizable (i.e. abnormal, described) studies got
    a crop, then "a crop exists" would itself announce abnormality and inflate the
    oracle far beyond a localization effect. So every study -- normals and
    non-localizable abnormals included -- receives exactly one crop, with its zone
    drawn from the MARGINAL distribution of the genuine zones. Crop presence, crop
    count and the zone distribution therefore carry no signal; only correctness
    differs. Seeded, so the assignment is reproducible."""
    import random
    from collections import Counter
    genuine = {}
    for uid, mesh in uid_mesh_pairs:
        z = mesh_zone(mesh)
        if z:
            genuine[uid] = z
    counts = Counter(genuine.values())
    if not counts:
        counts = Counter({z: 1 for z in ZONES})
    zones, weights = list(counts), [counts[z] for z in counts]
    rng = random.Random(seed)
    out = {}
    for uid, _ in uid_mesh_pairs:
        if uid in genuine:
            out[uid] = (genuine[uid], "localizable")
        else:
            out[uid] = (rng.choices(zones, weights=weights, k=1)[0], "assigned")
    return out


def _render_degraded(data: bytes, kind: str, param, seed: int) -> bytes:
    """Apply a controlled degradation to already-rendered bytes.

    Operating on the rendered baseline bytes (not the native file) guarantees the
    degraded arms differ from the baseline in exactly one respect: the degradation.
      blur     — Gaussian blur, radius `param`; removes fine detail, keeps layout.
      scramble — shuffle a `param`x`param` tile grid with a fixed seed; preserves
                 local texture and the global histogram while destroying anatomy.
      blank    — a uniform mid-grey field of the same size: the information floor.
    """
    from PIL import Image, ImageFilter
    import io as _io, random as _random
    with Image.open(_io.BytesIO(data)) as im:
        im = im.convert("L")
        if kind == "blur":
            im = im.filter(ImageFilter.GaussianBlur(radius=param))
        elif kind == "scramble":
            n = int(param)
            W, H = im.size
            tw, th = W // n, H // n
            tiles = [im.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
                     for r in range(n) for c in range(n)]
            _random.Random(seed).shuffle(tiles)
            out = Image.new("L", (tw * n, th * n))
            for i, t in enumerate(tiles):
                out.paste(t, ((i % n) * tw, (i // n) * th))
            im = out
        elif kind == "blank":
            im = Image.new("L", im.size, color=128)
        else:
            raise ValueError(f"unknown degradation {kind!r}")
        buf = _io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def assign_wrong_zones(uid_mesh_pairs, seed: int) -> dict:
    """-> {uid: (wrong_zone, "wrong")} for studies that HAVE a true zone.

    The wrong zone is drawn from the MARGINAL distribution of true zones, rejecting
    the study's own true zone. Sampling uniformly over the other five zones would
    have been simpler but introduces a confound: true zones here are ~60% basal, so
    a uniform wrong zone lands disproportionately on apices, and any observed harm
    could reflect "pointed at a quiet region" rather than "pointed incorrectly".
    Marginal-matched sampling makes the wrong-zone distribution resemble the true-
    zone distribution, so anatomy is held roughly constant and only correctness
    varies. Seeded, so the assignment is reproducible."""
    import random
    from collections import Counter
    rng = random.Random(seed)
    truth = {uid: mesh_zone(mesh) for uid, mesh in uid_mesh_pairs}
    truth = {u: z for u, z in truth.items() if z is not None}
    counts = Counter(truth.values())
    zones, weights = list(counts), [counts[z] for z in counts]
    out = {}
    for uid, z in truth.items():
        for _ in range(200):                       # reject the true zone
            w = rng.choices(zones, weights=weights, k=1)[0]
            if w != z:
                out[uid] = (w, "wrong")
                break
        else:                                      # degenerate fallback
            out[uid] = (rng.choice([w for w in ZONES if w != z]), "wrong")
    return out


def _render_crop(src: str, zone, max_side: int) -> bytes:
    """Crop from the NATIVE image, then downscale -- the order matters: cropping
    first is what buys effective resolution. Same convert('L') + PNG encode as
    every other part, so the crops differ from the full view only in framing."""
    from PIL import Image
    import io as _io
    with Image.open(src) as im:
        box = zone_box(zone, im.size[0], im.size[1])
        c = im.crop(box)
        if max(c.size) > max_side:
            sc = max_side / max(c.size)
            c = c.resize((int(c.size[0] * sc), int(c.size[1] * sc)))
        buf = _io.BytesIO()
        c.convert("L").save(buf, format="PNG")
        return buf.getvalue()

IMPLEMENTED_VARIANTS = tuple(VARIANT_SPECS)  # extended by the Exp-2 loop above

MANIFEST_COLUMNS = ["uid", "variant", "part_idx", "kind", "zone", "zone_source",
                    "path", "width", "height", "sha256", "src_path"]


def _head_existing(paths) -> list[str]:
    """Mirror `GeminiBackend.generate`'s selection EXACTLY: take paths[:1], and
    drop it when the file is missing. Matching this is what makes arm A a true
    reproduction of the current pipeline rather than an approximation of it."""
    return [p for p in (paths or [])[:1] if p and os.path.exists(p)]


def _dims(data: bytes) -> tuple[int, int]:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        return im.size


def variant_dir(out_root: str, variant: str) -> str:
    return os.path.join(out_root, variant)


def render_variant(row, variant: str, out_root: str, zone=None,
                   zone_source: str = "") -> list[dict]:
    """Render one study's image parts for `variant`; return manifest rows.

    Part order is the order they will be sent to the model: frontal, then lateral.
    """
    if variant not in IMPLEMENTED_VARIANTS:
        raise ValueError(
            f"variant {variant!r} is not implemented yet (have: {IMPLEMENTED_VARIANTS}). "
            "Experiment 1 builds one condition at a time.")

    uid = row["uid"]
    d = variant_dir(out_root, variant)
    os.makedirs(d, exist_ok=True)

    spec = VARIANT_SPECS[variant]
    if spec["lateral"]:
        sources = [("frontal", row.get("frontal_paths")),
                   ("lateral", row.get("lateral_paths"))]
    elif _head_existing(row.get("frontal_paths")):
        sources = [("frontal", row.get("frontal_paths"))]
    else:
        # Single-image design (required for resolution to reach the model at all).
        # 32/754 test studies have NO frontal view; fall back to their lateral so the
        # locked split stays whole at n=754 and every study still sends exactly ONE
        # image. The substitution is identical in both arms, so it cannot bias the
        # resolution contrast -- and dropping those studies would silently shrink the
        # pre-registered test set instead.
        sources = [("lateral", row.get("lateral_paths"))]

    parts: list[dict] = []
    for kind, plist in sources:
        for src in _head_existing(plist):
            # Reuse — not re-implement — the live encoder, so arm A is byte-exact.
            data, _mime = _load_image_bytes(src, spec["max_side"])
            if spec.get("degrade"):
                dk, dp = spec["degrade"]
                data = _render_degraded(data, dk, dp, seed=int(uid))
            dst = os.path.join(d, f"{uid}_{kind}.png")
            with open(dst, "wb") as f:
                f.write(data)
            w, h = _dims(data)
            parts.append({
                "uid": uid, "variant": variant, "part_idx": len(parts),
                "kind": kind, "zone": "", "zone_source": "", "path": dst,
                "width": w, "height": h,
                "sha256": hashlib.sha256(data).hexdigest(), "src_path": src,
            })

    if spec["crops"] == "oracle" and parts:
        if zone is None:
            raise ValueError("oracle variant needs a zone (see assign_oracle_zones)")
        full_src = parts[0]["src_path"]
        data = _render_crop(full_src, zone, spec["max_side"])
        zname = f"{zone[1]}_{zone[0]}"
        dst = os.path.join(d, f"{uid}_oracle_{zname}.png")
        with open(dst, "wb") as f:
            f.write(data)
        w, h = _dims(data)
        parts.append({
            "uid": uid, "variant": variant, "part_idx": len(parts),
            "kind": "crop", "zone": zname, "zone_source": zone_source, "path": dst,
            "width": w, "height": h,
            "sha256": hashlib.sha256(data).hexdigest(), "src_path": full_src,
        })

    if spec["crops"] == "all6" and parts:
        full_src = parts[0]["src_path"]          # crop the view we actually sent
        for zone in ZONES:
            data = _render_crop(full_src, zone, spec["max_side"])
            zname = f"{zone[1]}_{zone[0]}"
            dst = os.path.join(d, f"{uid}_zone_{zname}.png")
            with open(dst, "wb") as f:
                f.write(data)
            w, h = _dims(data)
            parts.append({
                "uid": uid, "variant": variant, "part_idx": len(parts),
                "kind": "crop", "zone": zname, "zone_source": "grid", "path": dst,
                "width": w, "height": h,
                "sha256": hashlib.sha256(data).hexdigest(), "src_path": full_src,
            })
    return parts


def manifest_path(out_dir: str, variant: str) -> str:
    """One manifest per variant, so building a later arm cannot clobber an
    earlier one's provenance."""
    return os.path.join(out_dir, f"exp1_image_manifest_{variant}.csv")


def load_manifest(out_dir: str, variant: str) -> dict:
    """-> {uid: {"frontal": [...], "lateral": [...], "captions": [...]}} in part order.

    Crops ride the SAME stream as the frontal view (they are views of it), so they
    land in "frontal" after the full image; `captions` is index-aligned to it and
    is empty-string for the uncaptioned full view."""
    import pandas as pd
    mp = manifest_path(out_dir, variant)
    if not os.path.exists(mp):
        raise SystemExit(f"missing image manifest {mp} — run 37_build_image_variants.py "
                         f"--variant {variant} first")
    m = pd.read_csv(mp).sort_values(["uid", "part_idx"])
    by_uid: dict = {}
    for r in m.itertuples():
        slot = by_uid.setdefault(r.uid, {"frontal": [], "lateral": [], "captions": []})
        if r.kind == "lateral":
            slot["lateral"].append(r.path)
        else:
            slot["frontal"].append(r.path)
            zone = "" if not isinstance(r.zone, str) or not r.zone else r.zone
            slot["captions"].append(
                zone_caption(tuple(zone.split("_")[::-1])) if zone else "")
    return by_uid


def contact_sheet(parts: list[dict], out_path: str, thumb: int = 300,
                  max_studies: int = 6) -> str | None:
    """Compose a labelled grid of the first `max_studies` studies' parts for
    eyeball QA. This is the check that catches a wrong crop box or a mirrored
    laterality — things no assertion on counts or bytes can see."""
    from PIL import Image, ImageDraw

    by_uid: dict = {}
    for p in parts:
        by_uid.setdefault(p["uid"], []).append(p)
    uids = list(by_uid)[:max_studies]
    if not uids:
        return None
    ncol = max(len(by_uid[u]) for u in uids)
    pad, label_h = 6, 14
    W = ncol * (thumb + pad) + pad
    H = len(uids) * (thumb + label_h + pad) + pad
    sheet = Image.new("L", (W, H), color=255)
    draw = ImageDraw.Draw(sheet)

    for r, uid in enumerate(uids):
        y = pad + r * (thumb + label_h + pad)
        for c, part in enumerate(sorted(by_uid[uid], key=lambda z: z["part_idx"])):
            x = pad + c * (thumb + pad)
            with Image.open(part["path"]) as im:
                im = im.convert("L")
                im.thumbnail((thumb, thumb))
                sheet.paste(im, (x, y + label_h))
            tag = f'{uid} {part["kind"]}' + (f' [{part["zone"]}]' if part["zone"] else "")
            draw.text((x, y + 2), f'{tag} {part["width"]}x{part["height"]}', fill=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    return out_path
