"""Load and join the IU X-Ray (OpenI) archive into a study-level table.

Join key is `uid`, which links:
  - indiana_reports.csv   (one row per study: findings, impression, MeSH, ...)
  - indiana_projections.csv (one row per image: uid -> filename -> projection)
  - images/images_normalized/<filename>

Unit of analysis is the STUDY (uid), not the image. Each study aggregates its
frontal and lateral image paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Paths:
    reports_csv: str
    proj_csv: str
    images_dir: str


def _clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_studies(paths: Paths, require_findings: bool = True) -> pd.DataFrame:
    """Return one row per study with report text + list of image paths.

    Columns: uid, indication, comparison, findings, impression, mesh, problems,
             report_text, frontal_paths, lateral_paths, n_images, has_frontal.
    """
    reports = pd.read_csv(paths.reports_csv)
    proj = pd.read_csv(paths.proj_csv)

    reports = reports.rename(columns={"MeSH": "mesh", "Problems": "problems"})
    for col in ["indication", "comparison", "findings", "impression", "mesh", "problems"]:
        reports[col] = reports[col].map(_clean_text)

    # report_text is the reference "evidence source": findings + impression.
    reports["report_text"] = (reports["findings"] + " " + reports["impression"]).str.strip()

    # Attach image paths per projection.
    def paths_for(uid: int, projection: str) -> list[str]:
        rows = proj[(proj.uid == uid) & (proj.projection == projection)]
        out = []
        for fn in rows.filename:
            p = os.path.join(paths.images_dir, fn)
            out.append(p)
        return out

    frontal, lateral, n_img = [], [], []
    proj_by_uid = {uid: g for uid, g in proj.groupby("uid")}
    for uid in reports.uid:
        g = proj_by_uid.get(uid)
        if g is None:
            frontal.append([]); lateral.append([]); n_img.append(0); continue
        f = [os.path.join(paths.images_dir, fn) for fn in g[g.projection == "Frontal"].filename]
        l = [os.path.join(paths.images_dir, fn) for fn in g[g.projection == "Lateral"].filename]
        frontal.append(f); lateral.append(l); n_img.append(len(g))

    reports["frontal_paths"] = frontal
    reports["lateral_paths"] = lateral
    reports["n_images"] = n_img
    reports["has_frontal"] = reports["frontal_paths"].map(len) > 0

    if require_findings:
        keep = (reports["findings"] != "") | (reports["impression"] != "")
        reports = reports[keep].copy()

    # Studies with at least one image are usable for Design A (vision).
    reports = reports[reports["n_images"] > 0].copy()
    reports = reports.reset_index(drop=True)
    return reports


def xxxx_density(text: str) -> float:
    """Fraction of tokens that are the IU de-identification scrub token XXXX.

    High density => report verification is unreliable (protocol §3.3, §13).
    """
    toks = text.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if "XXXX" in t) / len(toks)


if __name__ == "__main__":
    import sys, yaml
    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"))
    p = cfg["paths"]
    df = load_studies(Paths(p["reports_csv"], p["proj_csv"], p["images_dir"]),
                      require_findings=cfg["dataset"]["require_findings"])
    print(f"usable studies: {len(df)}")
    print(f"with frontal:   {df.has_frontal.sum()}")
    print(df[["uid", "n_images", "has_frontal"]].head())
