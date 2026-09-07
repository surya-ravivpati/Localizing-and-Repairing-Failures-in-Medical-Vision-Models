"""Shared import bootstrap + config loader for scripts."""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # pipeline/
sys.path.insert(0, ROOT)


def _load_dotenv():
    """Load KEY=VALUE lines from pipeline/.env into os.environ (no dependency).
    Keeps API keys out of the chat transcript and out of git (.env is ignored)."""
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def load_cfg(path=None):
    path = path or os.path.join(ROOT, "configs", "config.yaml")
    cfg = yaml.safe_load(open(path))
    # resolve paths relative to pipeline/ dir
    for k, v in cfg["paths"].items():
        if isinstance(v, str) and v.startswith(".."):
            cfg["paths"][k] = os.path.normpath(os.path.join(ROOT, v))
    cfg["paths"]["out_dir"] = os.path.join(ROOT, cfg["paths"]["out_dir"])
    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)
    return cfg
