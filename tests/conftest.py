from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_operator_path() -> Path:
    return FIXTURES_DIR / "sample_operator.md"


@pytest.fixture
def minimal_operator_path() -> Path:
    return FIXTURES_DIR / "minimal_operator.md"


@pytest.fixture
def operators_dir() -> Path:
    return Path("operators")
