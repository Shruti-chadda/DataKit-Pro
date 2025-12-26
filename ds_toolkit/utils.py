"""
ds_toolkit/utils.py

Utility helpers for DataKit Pro:
- load/save pipelines and models
- save report assets and create a zip for download
- simple CSV loader helper
- provide path to the uploaded file (if present in workspace)
"""

import os
import json
import joblib
import zipfile
import io
from typing import Any, Dict, List, Optional

import pandas as pd


# -------------------------
# Paths & constants
# -------------------------
# If a user-provided file was uploaded during this session and stored
# in the workspace, you can return its path using get_uploaded_file_path().
# NOTE: This path is environment-specific. Replace if needed.
UPLOADED_FILE_PATH = "/mnt/data/0ed660b0-7cd3-4005-b947-0b61f10aa56d.png"


# -------------------------
# File / CSV helpers
# -------------------------
def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV into a DataFrame (thin wrapper around pandas.read_csv)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)


# -------------------------
# Model persistence
# -------------------------
def save_pipeline(obj: Any, path: str) -> None:
    """Save a model or pipeline object to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_pipeline(path: str) -> Any:
    """Load a joblib-saved model/pipeline."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


# -------------------------
# Report & export helpers
# -------------------------
def write_json(obj: Dict, path: str) -> None:
    """Write a dictionary as pretty JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def create_report_zip(file_entries: List[Dict[str, bytes]], zip_path: str) -> None:
    """
    Create a zip archive containing report assets.

    file_entries: list of dicts with {'name': 'metrics.json', 'bytes': b'...'}
    zip_path: output zip path
    """
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for entry in file_entries:
            z.writestr(entry["name"], entry["bytes"])


def build_standard_report_assets(
    metrics: Dict,
    images: Optional[Dict[str, bytes]] = None,
    cleaned_csv_bytes: Optional[bytes] = None,
) -> List[Dict[str, bytes]]:
    """
    Build a list of file entries for the report zip.
    - metrics: dict (will be saved as metrics.json)
    - images: dict of name->bytes (pngs)
    - cleaned_csv_bytes: optional CSV bytes for cleaned dataset
    """
    entries = []
    entries.append({"name": "metrics.json", "bytes": json.dumps(metrics, indent=2).encode("utf-8")})
    if images:
        for name, b in images.items():
            entries.append({"name": f"images/{name}", "bytes": b})
    if cleaned_csv_bytes:
        entries.append({"name": "cleaned_dataset.csv", "bytes": cleaned_csv_bytes})
    return entries


# -------------------------
# Convenience: return uploaded file path (developer note)
# -------------------------
def get_uploaded_file_path() -> Optional[str]:
    """
    Return the local path of a previously uploaded file if present.
    This path points to a file inside the environment (e.g. /mnt/data/...).
    If no file is available, returns None.
    """
    path = UPLOADED_FILE_PATH
    return path if os.path.exists(path) else None
