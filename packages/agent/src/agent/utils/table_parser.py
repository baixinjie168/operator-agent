"""HTML table parser for direct column extraction.

Extracts shape, dtype, dformat and is_support_discontinuous directly from
HTML parameter tables, bypassing LLM inference entirely.

Uses only the standard library HTMLParser - no external dependencies.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any

# Column header keywords (normalised for comparison)
_SHAPE_NORM = {"维度(shape)", "维度（shape）", "维度"}
_DTYPE_NORM = {"数据类型"}
_DFORMAT_NORM = {"数据格式"}
_DISC_NORM = {"非连续tensor", "非连续 tensor"}
_DESC_NORM = {"描述", "参数描述"}
_USAGE_NORM = {"使用说明", "使用限制"}
_DIRECTION_NORM = {"输入/输出", "输入／输出", "方向"}

_HEADER_MAP: list[tuple[set[str], str]] = [
    (_SHAPE_NORM, "shape"),
    (_DTYPE_NORM, "dtype_desc"),
    (_DFORMAT_NORM, "dformat_desc"),
    (_DISC_NORM, "is_support_discontinuous"),
    (_DESC_NORM, "param_desc"),
    (_USAGE_NORM, "usage_notes"),
    (_DIRECTION_NORM, "direction"),
]

_SUPPORTED_RE = re.compile(r"[√✓✔]|(?<!不)支持")
_NOT_SUPPORTED_RE = re.compile(r"[×✗]|不支持")


class _Cell:
    """Accumulates text content inside a <td> / <th>."""
    __slots__ = ("text", "rowspan", "colspan", "is_header", "raw_parts")

    def __init__(self, is_header: bool = False, rowspan: int = 1, colspan: int = 1) -> None:
        self.text: list[str] = []
        self.rowspan = rowspan
        self.colspan = colspan
        self.is_header = is_header
        self.raw_parts: list[str] = []  # raw HTML fragments for platform-tag extraction

    def value(self) -> str:
        return " ".join("".join(self.text).split()).strip()

    def raw_html(self) -> str:
        """Return the raw HTML content accumulated inside this cell."""
        return "".join(self.raw_parts)


class _HTMLTableParser(HTMLParser):
    """Parse HTML into tables with rowspan/colspan expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self.raw_tables: list[list[list[str]]] = []  # parallel: raw HTML per cell
        self._current_table: list[list[_Cell]] = []
        self._current_row: list[_Cell] = []
        self._current_cell: _Cell | None = None
        self._in_table = False
        self._in_cell = False  # track whether we're inside td/th
        self._header_row: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            self._current_table = []
            self._header_row = []
        elif tag == "tr" and self._in_table:
            self._current_row = []
        elif tag in ("th", "td") and self._in_table:
            attr_dict = dict(attrs)
            rowspan = int(attr_dict.get("rowspan", "1") or "1")
            colspan = int(attr_dict.get("colspan", "1") or "1")
            self._current_cell = _Cell(
                is_header=(tag == "th"), rowspan=rowspan, colspan=colspan
            )
            self._in_cell = True
        # Track raw HTML for cell content
        if self._in_cell and self._current_cell is not None:
            raw = self.get_starttag_text() or f"<{tag}>"
            self._current_cell.raw_parts.append(raw)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        # Track raw HTML for cell content (before closing td/th)
        if self._in_cell and self._current_cell is not None:
            self._current_cell.raw_parts.append(f"</{tag}>")
        if tag in ("th", "td") and self._in_table and self._current_cell is not None:
            for _ in range(self._current_cell.colspan):
                self._current_row.append(self._current_cell)
            if self._current_cell.is_header and not self._header_row:
                for _ in range(self._current_cell.colspan):
                    self._header_row.append(self._current_cell.value())
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._in_table:
            self._current_table.append(self._current_row)
            self._current_row = []
        elif tag == "table" and self._in_table:
            grid = self._expand_rowspan(self._current_table)
            str_grid = [[cell.value() for cell in row] for row in grid]
            raw_grid = [[cell.raw_html() for cell in row] for row in grid]
            self.tables.append(str_grid)
            self.raw_tables.append(raw_grid)
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.text.append(data)
            # Also append text data to raw_parts so that raw_html()
            # includes the text content between tags (e.g. the platform
            # name inside <term>...</term>).
            if self._in_cell:
                self._current_cell.raw_parts.append(data)

    @staticmethod
    def _expand_rowspan(rows: list[list[_Cell]]) -> list[list[_Cell]]:
        """Expand rowspan by copying cell references into subsequent rows."""
        if not rows:
            return rows
        max_cols = max(len(r) for r in rows) if rows else 0
        result: list[list[_Cell | None]] = [[None] * max_cols for _ in rows]
        for r_idx, row in enumerate(rows):
            c_idx = 0
            for cell in row:
                while c_idx < max_cols and result[r_idx][c_idx] is not None:
                    c_idx += 1
                if c_idx >= max_cols:
                    break
                for dr in range(cell.rowspan):
                    rr = r_idx + dr
                    if rr >= len(result):
                        result.append([None] * max_cols)
                    result[rr][c_idx] = cell
                c_idx += 1
        empty = _Cell()
        return [[cell or empty for cell in row] for row in result]


def parse_html_tables_with_raw(
    content: str,
) -> tuple[list[list[list[str]]], list[list[list[str]]]]:
    """Parse all html tables, returning (text_grids, raw_html_grids).

    text_grids: list of tables, each a list of rows, each a list of cell text.
    raw_html_grids: parallel structure with raw HTML per cell.
    """
    parser = _HTMLTableParser()
    try:
        parser.feed(content)
    except Exception:
        return [], []
    return parser.tables, parser.raw_tables


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


# ---------------------------------------------------------------------------
# Parameter name matching helpers
# ---------------------------------------------------------------------------

# Matches type annotation suffix in table cells:
#   'x（aclTensor*）' → 'x'
#   'innerPrecise（int64_t）' → 'innerPrecise'
#   'activation（char*）' → 'activation'
_TYPE_ANNOTATION_RE = re.compile(r"\s*[（(]\s*[a-zA-Z][a-zA-Z0-9_*\s]*[）)]\s*$")


def _strip_type_annotation(cell: str) -> str:
    """Strip type annotation suffix from a table cell value.

    'x（aclTensor*）' → 'x'
    'innerPrecise（int64_t）' → 'innerPrecise'
    'activation（char*）' → 'activation'
    """
    return _TYPE_ANNOTATION_RE.sub("", cell).strip()


def _match_param_name(cell: str, param_name: str) -> bool:
    """Check whether *cell* (a table name-cell) matches *param_name*.

    Handles three cases:
    1. Exact match: ``'x' == 'x'``
    2. Type annotation suffix: ``'x（aclTensor*）'`` matches ``'x'``
    3. Legacy star-suffix: ``'self*'`` matches ``'self'``
    """
    cell = cell.strip()
    if cell == param_name:
        return True
    if _strip_type_annotation(cell) == param_name:
        return True
    if cell.replace("*", "").strip() == param_name:
        return True
    return False


def detect_table_columns(header_row: list[str]) -> dict[str, int]:
    """Map canonical field names to column indices from a header row."""
    mapping: dict[str, int] = {}
    for col_idx, header_text in enumerate(header_row):
        norm = _normalise(header_text)
        for header_set, field_name in _HEADER_MAP:
            if field_name in mapping:
                continue
            for h in header_set:
                if _normalise(h) == norm or _normalise(h) in norm:
                    mapping[field_name] = col_idx
                    break
    return mapping


def is_table_form(header_row: list[str]) -> bool:
    """Return True if header row has at least 3 of the 4 target columns."""
    return len(detect_table_columns(header_row)) >= 3


def _clean_cell_value(value: str) -> str:
    v = value.strip()
    if v in {"-", "—", "–", "－"}:
        return ""
    return v


# ---------------------------------------------------------------------------
# Relative-reference detection
# ---------------------------------------------------------------------------
# Some table cells contain cross-references like "与self一致", "同input一
# 致", "与`self`相同" instead of concrete values.  These should NOT be
# treated as valid dtype / shape / dformat values — downstream LLM nodes
# need to resolve them.

_RELATIVE_REF_RE = re.compile(
    r"^(?:与|同|和|跟)"          # leading preposition
    r".{1,20}"                   # target param name (possibly in backticks)
    r"(?:一致|相同|一样|保持一致|保持一致|同)$",  # trailing comparator
)


def _is_relative_ref(value: str) -> bool:
    """Return True if *value* is a cross-reference like '与self一致'.

    Such values appear in dtype / shape / dformat columns when a parameter
    inherits its properties from another parameter.  They are not directly
    usable and must be resolved by the LLM extract nodes.
    """
    v = value.strip()
    if not v:
        return False
    # Strip backtick quoting: 与`self`一致 → 与self一致
    v = v.replace("`", "")
    return bool(_RELATIVE_REF_RE.match(v))


def _extract_direction(cell_value: str) -> str:
    """Normalize direction cell value to 'input' / 'output' / ''.

    Handles Chinese variants: 输入/输出/入参/出参/IN/OUT.
    """
    cell = cell_value.strip()
    if not cell or cell in {"-", "—", "–", "－"}:
        return ""
    lower = cell.lower()
    if "输入" in cell or "入参" in cell or lower == "in" or lower == "input":
        return "input"
    if "输出" in cell or "出参" in cell or lower == "out" or lower == "output":
        return "output"
    return ""


def _extract_discontinuous(cell_value: str, param_type: str) -> str:
    """Extract is_support_discontinuous from non-continuous Tensor column.

    Returns JSON string like '{"value": true, "src_text": "√"}'.
    """
    cell = cell_value.strip()
    is_tensor = "tensor" in param_type.lower()

    if not cell or cell in {"-", "—", "–", "－"}:
        if is_tensor:
            return json.dumps({"value": False, "src_text": ""}, ensure_ascii=False)
        return json.dumps({"value": "N/A", "src_text": ""}, ensure_ascii=False)

    if _SUPPORTED_RE.search(cell):
        return json.dumps({"value": True, "src_text": cell}, ensure_ascii=False)

    if _NOT_SUPPORTED_RE.search(cell):
        return json.dumps({"value": False, "src_text": cell}, ensure_ascii=False)

    # Unrecognised value: safe fallback
    if is_tensor:
        return json.dumps({"value": False, "src_text": cell}, ensure_ascii=False)
    return json.dumps({"value": "N/A", "src_text": cell}, ensure_ascii=False)


def find_param_name_column(header_row: list[str]) -> int:
    """Detect the parameter-name column index.  Falls back to 0."""
    name_headers = {"参数名", "参数名称", "parameter", "parameter name", "参数"}
    for idx, h in enumerate(header_row):
        norm = _normalise(h)
        for nh in name_headers:
            if _normalise(nh) == norm or _normalise(nh) in norm:
                return idx
    return 0


# ---------------------------------------------------------------------------
# Platform-aware JSON resolution
# ---------------------------------------------------------------------------

def resolve_platform_value(json_field: dict | str, platform: str) -> str:
    """Resolve a platform-aware JSON field to a specific platform's value.

    Priority: exact platform match → wildcard "*" → empty string.
    Also handles legacy flat text strings for backward compatibility.
    """
    if isinstance(json_field, str):
        if not json_field:
            return ""
        try:
            json_field = json.loads(json_field)
        except (json.JSONDecodeError, TypeError):
            return json_field  # legacy flat text
    if not isinstance(json_field, dict):
        return ""
    return json_field.get(platform) or json_field.get("*", "")


# ---------------------------------------------------------------------------
# Platform-tagged cell extraction
# ---------------------------------------------------------------------------
# Some operator docs use <li><term>PLATFORM</term>: VALUE</li> patterns
# inside table cells to specify per-platform dtype/shape/format.
# extract_platform_tagged_values() parses these patterns from raw HTML.

_PLATFORM_LI_RE = re.compile(
    r"<li[^>]*>\s*<term>([^<]+)</term>\s*[：:]\s*(.+?)\s*</li>",
    re.DOTALL | re.IGNORECASE,
)

# Inline <term>PLATFORM</term>：VALUE without <li> wrapper.
# Stops at the next <term>, <br>, </td>, or end-of-string.
# NOTE: deliberately does NOT stop at </li> — the VALUE itself may contain
# nested <ul><li>...</li><li>...</li></ul> lists (e.g. per-channel /
# per-group shape descriptions). Stopping at the first </li> would silently
# truncate the content after the first list item.
_PLATFORM_INLINE_RE = re.compile(
    r"<term>([^<]+)</term>\s*[：:]\s*(.+?)(?=<term>|<br\s*/?>|</td>|$)",
    re.DOTALL | re.IGNORECASE,
)


def extract_platform_tagged_values(raw_html_cell: str) -> dict[str, str] | None:
    """Parse platform-tagged values from raw HTML cell content.

    Supports two patterns:
    1. ``<li><term>PLATFORM</term>：VALUE</li>``  (list format, most common)
    2. ``<term>PLATFORM</term>：VALUE``            (inline format, no ``<li>``)

    Returns:
        ``{platform_name: value_text}`` if platform tags found.
        ``None`` if no platform tags (universal value — applies to all platforms).
    """
    if not raw_html_cell:
        return None
    # Pattern 1: <li><term>PLATFORM</term>：VALUE</li>
    matches = _PLATFORM_LI_RE.findall(raw_html_cell)
    if matches:
        return {platform.strip(): value.strip() for platform, value in matches}
    # Pattern 2: inline <term>PLATFORM</term>：VALUE (no <li>)
    matches = _PLATFORM_INLINE_RE.findall(raw_html_cell)
    if matches:
        return {platform.strip(): value.strip() for platform, value in matches}
    return None


# ---------------------------------------------------------------------------
# Unified JSON-format column extraction
# ---------------------------------------------------------------------------
# Replaces the two-step (extract_4_columns + extract_platform_attributes)
# with a single function that returns JSON {platform: value} for each field.

# Fields that support platform-aware JSON storage
_JSON_FIELDS = {"shape", "dtype_desc", "dformat_desc", "param_desc", "usage_notes"}


def _cell_to_json(
    text_value: str,
    raw_html: str,
    field: str,
    param_type: str = "",
) -> dict[str, str] | str | None:
    """Convert a single cell to JSON {platform: value} format.

    Returns:
        dict: Platform-keyed values (if platform tags found or plain value wrapped as {"*": value})
        str: For non-JSON fields (direction, is_support_discontinuous)
        None: If cell is empty / dash / relative ref
    """
    # Check for platform-tagged values first
    tagged = extract_platform_tagged_values(raw_html)
    if tagged:
        # Clean each platform value
        cleaned = {}
        for plat, val in tagged.items():
            cv = _clean_cell_value(val)
            if cv and not _is_relative_ref(cv):
                cleaned[plat] = cv
        return cleaned if cleaned else None

    # No platform tags → universal value
    clean = _clean_cell_value(text_value)
    if not clean:
        return None

    # For shape/dtype/dformat/param_desc/usage_notes: skip relative refs
    if field in _JSON_FIELDS and _is_relative_ref(clean):
        return None

    return {"*": clean}


def extract_columns_as_json(
    text_grid: list[list[str]],
    raw_grid: list[list[str]],
    col_map: dict[str, int],
    param_name: str,
    param_type: str,
    name_col_idx: int = 0,
) -> dict[str, Any]:
    """Find param_name in a parsed table and extract all columns as JSON format.

    Returns a dict with keys: shape, dtype_desc, dformat_desc, param_desc,
    usage_notes, direction, is_support_discontinuous.
    All value fields are JSON {platform: value} dicts.
    direction and is_support_discontinuous are plain strings/JSON strings.

    Returns empty dict if param_name not found.
    """
    result: dict[str, Any] = {}
    target_text_row: list[str] | None = None
    target_raw_row: list[str] | None = None

    for i, row in enumerate(text_grid):
        if name_col_idx < len(row):
            cell = row[name_col_idx].strip()
            if _match_param_name(cell, param_name):
                target_text_row = row
                target_raw_row = raw_grid[i] if i < len(raw_grid) else None
                break

    if target_text_row is None:
        return result

    # JSON fields: shape, dtype_desc, dformat_desc, param_desc, usage_notes
    for field in _JSON_FIELDS:
        if field not in col_map:
            continue
        idx = col_map[field]
        if idx >= len(target_text_row):
            continue
        text_val = target_text_row[idx]
        raw_val = target_raw_row[idx] if target_raw_row and idx < len(target_raw_row) else ""
        json_val = _cell_to_json(text_val, raw_val, field, param_type)
        if json_val:
            result[field] = json_val

    # is_support_discontinuous: special handling (returns JSON string)
    if "is_support_discontinuous" in col_map:
        idx = col_map["is_support_discontinuous"]
        if idx < len(target_text_row):
            result["is_support_discontinuous"] = _extract_discontinuous(
                target_text_row[idx], param_type
            )

    # direction: special handling (returns plain string)
    if "direction" in col_map:
        idx = col_map["direction"]
        if idx < len(target_text_row):
            direction = _extract_direction(target_text_row[idx])
            if direction:
                result["direction"] = direction

    return result
