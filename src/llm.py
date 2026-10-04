from __future__ import annotations

"""Unified LLM access layer.

Tự động chọn provider: OpenAI (nếu có key `sk-`) hoặc OpenCode Go
(OpenAI-compatible, key lấy từ auth.json). Dùng chung cho M5 enrichment,
pipeline generation và RAGAS judge.
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (LLM_ENABLED, LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY,
                    LLM_MODEL, EMBEDDING_MODEL)

# Session ID ổn định cho OpenCode Go routing/prompt caching.
_SESSION_ID = str(uuid.uuid4())
_HEADERS = {"x-opencode-session": _SESSION_ID, "User-Agent": "lab18-rag/1.0"}


def llm_enabled() -> bool:
    return LLM_ENABLED


def get_client():
    """Trả về OpenAI-compatible client cho provider đang dùng."""
    from openai import OpenAI

    kwargs: dict = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    if LLM_PROVIDER == "opencode-go":
        kwargs["default_headers"] = _HEADERS
    return OpenAI(**kwargs)


def chat(messages: list[dict], max_tokens: int = 400, temperature: float = 0.0) -> str:
    """Gọi chat completion, trả về text (rỗng nếu thất bại)."""
    client = get_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return (resp.choices[0].message.content or "").strip()


def get_ragas_llm():
    """LangChain LLM đã wrap cho RAGAS."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    kwargs: dict = {"model": LLM_MODEL, "api_key": LLM_API_KEY, "temperature": 0.0}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    if LLM_PROVIDER == "opencode-go":
        kwargs["default_headers"] = _HEADERS
    return LangchainLLMWrapper(ChatOpenAI(**kwargs))


def get_ragas_embeddings():
    """Local embeddings (bge-m3) đã wrap cho RAGAS — không cần endpoint embeddings."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))
