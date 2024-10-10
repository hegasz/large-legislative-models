import os
from pathlib import Path

from eztils import abspath, setup_path


class Globals:
    REPO_DIR = setup_path(Path(abspath()) / "..")
    DATA_ROOT = setup_path(os.getenv("DATA_ROOT") or REPO_DIR)
    RUN_DIR = LOG_DIR = Path()
