"""Shared configuration for Lab 18."""

import json
import os
from dotenv import load_dotenv

load_dotenv()


def _load_opencode_key() -> str:
    """Lấy API key OpenCode Go từ env hoặc auth.json của opencode."""
    env_key = os.getenv("OPENCODE_API_KEY", "")
    if env_key:
        return env_key
    candidates = [
        os.path.join(os.path.expanduser("~"), ".local", "share", "opencode", "auth.json"),
        os.path.join(os.path.expanduser("~"), ".config", "opencode", "auth.json"),
    ]
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for prov in ("opencode-go", "opencode"):
                entry = data.get(prov)
                if isinstance(entry, dict) and entry.get("key"):
                    return entry["key"]
        except Exception:
            continue
    return ""


# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Chỉ chấp nhận key OpenAI hợp lệ (bắt đầu bằng "sk-"). Các key lạ bị inject
# từ môi trường (vd key của tool khác) sẽ bị bỏ qua.
if not OPENAI_API_KEY.startswith("sk-"):
    OPENAI_API_KEY = ""

OPENCODE_API_KEY = _load_opencode_key()
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
OPENCODE_MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash")

# --- Unified LLM provider selection (OpenAI ưu tiên, fallback OpenCode Go) ---
if OPENAI_API_KEY:
    LLM_ENABLED = True
    LLM_PROVIDER = "openai"
    LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
    LLM_API_KEY = OPENAI_API_KEY
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
elif OPENCODE_API_KEY:
    LLM_ENABLED = True
    LLM_PROVIDER = "opencode-go"
    LLM_BASE_URL = OPENCODE_BASE_URL
    LLM_API_KEY = OPENCODE_API_KEY
    LLM_MODEL = OPENCODE_MODEL
else:
    LLM_ENABLED = False
    LLM_PROVIDER = None
    LLM_BASE_URL = None
    LLM_API_KEY = ""
    LLM_MODEL = ""

# Cho phép ép chạy offline (dùng cho unit tests để không gọi mạng).
if os.getenv("LAB18_OFFLINE", "").strip().lower() not in ("", "0", "false", "no"):
    LLM_ENABLED = False
    LLM_PROVIDER = None
    LLM_BASE_URL = None
    LLM_API_KEY = ""
    LLM_MODEL = ""

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
