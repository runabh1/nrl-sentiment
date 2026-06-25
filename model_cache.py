"""
NRL Sentinel — Model Persistence Layer
Saves trained models to disk so they load instantly on subsequent app starts.

Why this matters:
- Training all 5 models takes 60-120 seconds on first run
- Without persistence, every app restart re-trains from scratch
- With persistence, subsequent loads take < 2 seconds (just deserialise)
- Models are invalidated (re-trained) when data files change or after 24 hours
"""

import os
import pickle
import hashlib
import time
import pandas as pd
import numpy as np

MODELS_DIR = "trained_models"
METADATA_FILE = os.path.join(MODELS_DIR, "metadata.pkl")
MAX_AGE_HOURS = 24   # Re-train if model is older than this

os.makedirs(MODELS_DIR, exist_ok=True)


def _data_fingerprint(*dfs) -> str:
    """
    Create a short hash from the shapes + last-row values of all input dataframes.
    If data changes (new rows fetched), fingerprint changes and models are re-trained.
    """
    parts = []
    for df in dfs:
        if df is None or len(df) == 0:
            parts.append("empty")
            continue
        shape_str  = f"{df.shape}"
        tail_str   = str(df.iloc[-1].values[:5].tolist())
        parts.append(shape_str + tail_str)
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def _load_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_metadata(meta: dict):
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(meta, f)


def _model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, f"{name}.pkl")


def models_are_valid(crude_df, demand_df, sensor_df, project_df, pipeline_df) -> bool:
    """
    Returns True if saved models exist, are fresh (<24h), and match current data fingerprint.
    """
    meta = _load_metadata()
    if not meta:
        return False

    # Check age
    trained_at = meta.get("trained_at", 0)
    age_hours  = (time.time() - trained_at) / 3600
    if age_hours > MAX_AGE_HOURS:
        print(f"[MODEL CACHE] Models are {age_hours:.1f}h old — will re-train (max {MAX_AGE_HOURS}h)")
        return False

    # Check data fingerprint
    current_fp = _data_fingerprint(crude_df, demand_df, sensor_df, project_df, pipeline_df)
    saved_fp   = meta.get("data_fingerprint", "")
    if current_fp != saved_fp:
        print(f"[MODEL CACHE] Data has changed (fp {saved_fp} -> {current_fp}) — will re-train")
        return False

    # Check all model files exist
    for name in ["grm", "demand", "maintenance", "overrun", "pipeline"]:
        if not os.path.exists(_model_path(name)):
            print(f"[MODEL CACHE] {name}.pkl missing — will re-train")
            return False

    print(f"[MODEL CACHE] Loaded from cache (trained {age_hours:.1f}h ago, fp={saved_fp})")
    return True


def save_all_models(grm_r, demand_r, maint_r, overrun_r, pipeline_r,
                    crude_df, demand_df, sensor_df, project_df, pipeline_df):
    """Persist all model results to disk."""
    for name, result in [("grm", grm_r), ("demand", demand_r),
                          ("maintenance", maint_r), ("overrun", overrun_r),
                          ("pipeline", pipeline_r)]:
        path = _model_path(name)
        try:
            with open(path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[MODEL CACHE] Warning: could not save {name}: {e}")

    fp   = _data_fingerprint(crude_df, demand_df, sensor_df, project_df, pipeline_df)
    meta = {
        "trained_at":       time.time(),
        "data_fingerprint": fp,
        "model_versions": {
            "grm":         "v3",
            "demand":      "v2",
            "maintenance": "v2",
            "overrun":     "v2",
            "pipeline":    "v2",
        }
    }
    _save_metadata(meta)
    print(f"[MODEL CACHE] All models saved (fp={fp})")


def load_all_models():
    """Load all model results from disk. Returns tuple or None if any missing."""
    results = []
    for name in ["grm", "demand", "maintenance", "overrun", "pipeline"]:
        path = _model_path(name)
        try:
            with open(path, "rb") as f:
                results.append(pickle.load(f))
        except Exception as e:
            print(f"[MODEL CACHE] Could not load {name}: {e}")
            return None
    return tuple(results)


def clear_model_cache():
    """Force re-training on next app start."""
    import shutil
    if os.path.exists(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
        os.makedirs(MODELS_DIR, exist_ok=True)
    print("[MODEL CACHE] Cache cleared — models will re-train on next load")


def get_cache_info() -> dict:
    """Return human-readable cache status for display in the UI."""
    meta = _load_metadata()
    if not meta:
        return {"status": "No cache", "age_hours": None, "fingerprint": None}
    age = (time.time() - meta.get("trained_at", 0)) / 3600
    return {
        "status":      "Valid" if age < MAX_AGE_HOURS else "Stale",
        "age_hours":   round(age, 1),
        "fingerprint": meta.get("data_fingerprint", "unknown"),
        "trained_at":  pd.Timestamp(meta["trained_at"], unit="s").strftime("%Y-%m-%d %H:%M"),
    }
