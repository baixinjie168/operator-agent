"""Shared LLM factory for all pipeline nodes."""

from langchain_openai import ChatOpenAI

from agent.core.config import settings


def create_llm(*, temperature: float = 0.1) -> ChatOpenAI:
    """Create a ChatOpenAI instance with centralized config.

    All nodes should use this factory instead of constructing ChatOpenAI
    directly, so that max_tokens / api_key / model changes propagate
    automatically.
    """
    return ChatOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        model=settings.active_model,
        max_tokens=settings.llm_max_tokens,
        temperature=temperature,
    )