"""Public facade: ``TestCaseGenerator``.

Thin wrapper around :func:`agent.generators.generator.generate` that holds
the parsed ``GeneratorContext`` and seed for the lifetime of a generation
request.  The single underlying ``generate`` function is the only place
that needs to be replaced when the formal generation logic lands.

The public API of this module is what node / route / MCP tool callers
import; they don't need to know which sampling strategy is in use.
"""

from __future__ import annotations

import logging

from agent.generators.generator import generate
from shared.models.test_case import GeneratorContext, TestCaseRecord

logger = logging.getLogger(__name__)

DEFAULT_COUNT = 10
DEFAULT_SEED: int | None = 42


class TestCaseGenerator:
    """Generate test cases from a parsed ``GeneratorContext``.

    This is the **single entry point** used by node / route / MCP code.
    It is intentionally a thin wrapper: it holds the context + seed and
    delegates to ``agent.generators.generator.generate``.
    """

    __test__ = False  # not a pytest test class

    def __init__(
        self,
        context: GeneratorContext,
        *,
        seed: int | None = DEFAULT_SEED,
    ) -> None:
        self._context = context
        self._seed = seed
        logger.debug(
            "TestCaseGenerator: context=%s count_default=%d seed=%s",
            context.operator_name, DEFAULT_COUNT, seed,
        )

    def generate(self, count: int = DEFAULT_COUNT) -> list[TestCaseRecord]:
        """Return ``count`` test cases.

        Contract:
            generate(count) -> list[TestCaseRecord]

        Determinism:
            Same ``context`` + same ``seed`` ⇒ identical case list.
        """
        return generate(self._context, count=count, seed=self._seed)
