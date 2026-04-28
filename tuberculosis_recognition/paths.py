from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
SOURCE_DIR = PROJECT_ROOT / "source"
DATA_DIR = SOURCE_DIR / "data"
MODEL_DIR = SOURCE_DIR / "model"
EXAMPLES_DIR = DATA_DIR / "examples"
DATA_PARAMS_PATH = DATA_DIR / "data_params.json"
MODEL_WEIGHTS_PATH = MODEL_DIR / "model_weights.pth"
