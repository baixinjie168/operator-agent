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


@pytest.fixture
def sample_result_json() -> dict:
    """Minimal ``json_constraints`` dict used by the case-subgraph test suite.

    Mirrors the shape produced by ``assemble_result``: ``operator_name`` +
    per-platform ``inputs`` / ``outputs`` / ``constraints_in_parameters`` /
    ``dtype_support_description`` / ``format_support_description``.

    The shape candidate counts and dtype set are deliberately non-empty so
    ``case_init_static`` produces ``sampled_shapes >= 1`` and
    ``sampled_dtypes >= 1``. The fixture also carries the
    ``is_operator_param`` boolean that the formal generation code's
    Pydantic validator requires.
    """
    param_attrs_x1 = {
        "dtype": {"value": ["FLOAT32"], "src_text": ""},
        "dimensions": {"value": [[2, 8]], "src_text": "2-8"},
        "is_optional": {"value": False, "src_text": ""},
        "is_operator_param": {"value": True, "src_text": ""},
        "type": {"value": "const aclTensor", "src_text": ""},
        "format": {"value": "ND", "src_text": ""},
    }
    param_attrs_y1 = dict(param_attrs_x1)
    return {
        "operator_name": "aclnnSample",
        "product_support": ["P1"],
        "inputs": {
            "x1": {"P1": param_attrs_x1},
        },
        "outputs": {
            "y1": {"P1": param_attrs_y1},
        },
        "constraints_in_parameters": {
            "P1": [],
        },
        "dtype_support_description": {
            "P1": [{"x1": "FLOAT32", "y1": "FLOAT32"}],
        },
        "format_support_description": {
            "P1": [{"x1": "ND", "y1": "ND"}],
        },
    }
