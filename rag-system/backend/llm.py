"""
LLM factory — supports Groq and NVIDIA NIM only.

Embeddings run in-process via sentence-transformers (see ingestion.py).
Switch chat providers via LLM_PROVIDER env var.
"""
from __future__ import annotations
from .config import SETTINGS


def get_chat_llm(model: str | None = None, temperature: float | None = None,
                 num_predict: int | None = None):
    """Return a chat model based on the active provider."""
    model = model or SETTINGS.llm_model
    temperature = SETTINGS.llm_temperature if temperature is None else temperature
    provider = SETTINGS.llm_provider

    if provider == "groq":
        from langchain_groq import ChatGroq
        if not SETTINGS.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        return ChatGroq(
            api_key=SETTINGS.groq_api_key,
            model=model,
            temperature=temperature,
            max_tokens=num_predict or 2048,
        )

    if provider == "nvidia":
        from langchain_openai import ChatOpenAI
        if not SETTINGS.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY not set in .env")
        return ChatOpenAI(
            api_key=SETTINGS.nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            model=model,
            temperature=temperature,
            max_tokens=num_predict or 2048,
        )

    raise RuntimeError(
        f"Unknown LLM_PROVIDER: '{provider}'. Use 'groq' or 'nvidia'."
    )


def get_rewrite_llm():
    """Smaller/faster model for cheap tasks like query rewriting."""
    return get_chat_llm(
        model=SETTINGS.rewrite_model,
        temperature=0.0,
        num_predict=200,
    )