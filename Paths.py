# paths.py
from pathlib import Path
import sys, os

def app_base() -> Path:
    if getattr(sys, "frozen", False):            # 被 PyInstaller 包過
        if hasattr(sys, "_MEIPASS"):             # onefile 展開後的臨時路徑
            return Path(sys._MEIPASS)
        return Path(sys.executable).parent       # onedir：exe 所在資料夾
    return Path(__file__).resolve().parent       # 開發環境：.py 所在目錄

BASE = app_base()
MODELS_DIR = BASE / "models"                     # 權重/Tokenizer 放這邊
DATA_DIR   = BASE / "data"                       # 若有 faiss/index/metadata

# 完全離線（Transformers / SentenceTransformers）
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 有裝 huggingface-hub 時生效


def ensure_dir(p: Path, hint: str = ""):
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}\n{hint}")

def rp(rel: str) -> Path:
    return BASE / rel
