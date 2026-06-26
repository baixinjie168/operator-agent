"""Shared LLM factory for all pipeline nodes."""

from langchain_openai import ChatOpenAI

from agent.core.config import LLMProvider, settings


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


def create_constraint_check_llm(*, temperature: float = 0.1) -> ChatOpenAI:
    """Create LLM for constraint checking (separate model from generation).

    Uses ``constraint_check_llm_provider`` and ``constraint_check_model``
    from settings, which default to DeepSeek for stronger reasoning.
    This avoids self-evaluation bias (generation model judging its own output).
    """
    import logging
    _log = logging.getLogger(__name__)

    provider = settings.constraint_check_llm_provider
    api_key = settings.get_provider_api_key(provider)
    base_url = settings.get_provider_base_url(provider)
    model = settings.constraint_check_model or settings.get_provider_model(provider)
    max_tokens = settings.constraint_check_max_tokens

    _log.info(
        "ConstraintCheckLLM: provider=%s, model=%s, base_url=%s, max_tokens=%d",
        provider, model, base_url, max_tokens,
    )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=1800,      # 30 minutes per request (large input + HTML generation)
        max_retries=3,     # retry up to 3 times on transient errors
    )