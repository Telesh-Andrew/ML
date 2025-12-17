from pathlib import Path


def get_project_root() -> Path:
    """Return absolute path to the project root (repository root).

    We assume this file lives under `/app/src/app` in Docker image
    and under `<project_root>/src/app` locally.
    """
    # /path/to/project/src/app/config.py -> /path/to/project
    # /app/src/app/config.py              -> /app
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = get_project_root()
MODEL_DIR: Path = PROJECT_ROOT / "artifacts" / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"

MODEL_JOBLIB_PATH: Path = MODEL_DIR / "lightgbm_smape_11.74.joblib"
MODEL_META_PATH: Path = MODEL_DIR / "lightgbm_smape_11.74.json"
TEST_FEATURES_FILE: Path = DATA_DIR / "processed" / "test_features_cleaned.csv"

