import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR_ENV = "SPATH_WORK_DIR"


def get_work_dir(value=None):
    root = value or os.environ.get(WORK_DIR_ENV)
    if root:
        return Path(root).expanduser().resolve()
    return REPO_ROOT


def resolve_path(path, base_dir=None):
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir or get_work_dir()) / path
