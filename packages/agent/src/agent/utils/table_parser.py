"""HTML table parser for direct column extraction.

Extracts shape, dtype, dformat and is_support_discontinuous directly from
HTML parameter tables, bypassing LLM inference entirely.

Uses only the standard library HTMLParser - no external dependencies.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

# Column header keywords (normalised for comparison)
_SHAPE_NORM = {"维度(shape)", "维度（shape）", "维度"}
_DTYPE_NORM = {"数据类型"}
_DFORMAT_NORM = {"数据格式"}
_DISC_NORM = {"非连续tensor", "非连续 tensor"}
_DESC_NORM = {"描述", "参数描述", "说明"}
_DIRECTION_NORM = {"输入/输出", "输入／输出", "方向"}

_HEADER_MAP: list[tuple[set[str], str]] = [
    (_SHAPE_NORM, "shape"),
    (_DTYPE_NORM, "dtype_desc"),
    (_DFORMAT_NORM, "dformat_desc"),
    (_DISC_NORM, "is_support_discontinuous"),
    (_DESC_NORM, "param_desc"),
    (_DIRECTION_NORM, "direction"),
]

_SUPPORTED_RE = re.compile(r"[√✓✔]|(?<!不)支持")
_NOT_SUPPORTED_RE = re.compile(r"[×✗]|不支持")


class _Cell:
    """Accumulates text content inside a <td> / <th>."""
    __slots__ = ("text", "rowspan", "colspan", "is_header")

    def __init__(self, is_header: bool = False, rowspan: int = 1, colspan: int = 1) -> None:
        self.text: list[str] = []
        self.rowspan = rowspan
        self.colspan = colspan
        self.is_header = is_header

    def value(self) -> str:
        return " ".join("".join(self.text).split()).strip()


class _HTMLTableParser(HTMLParser):
    """Parse HTML into tables with rowspan/colspan expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[_Cell]] = []
        self._current_row: list[_Cell] = []
        self._current_cell: _Cell | None = None
        self._in_table = False
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

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("th", "td") and self._in_table and self._current_cell is not None:
            for _ in range(self._current_cell.colspan):
                self._current_row.append(self._current_cell)
            if self._current_cell.is_header and not self._header_row:
                for _ in range(self._current_cell.colspan):
                    self._header_row.append(self._current_cell.value())
            self._current_cell = None
        elif tag == "tr" and self._in_table:
            self._current_table.append(self._current_row)
            self._current_row = []
        elif tag == "table" and self._in_table:
            grid = self._expand_rowspan(self._current_table)
            str_grid = [[cell.value() for cell in row] for row in grid]
            self.tables.append(str_grid)
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.text.append(data)

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


def parse_html_tables(content: str) -> list[list[list[str]]]:
    """Parse all HTML tables from content, returning list of grids."""
    parser = _HTMLTableParser()
    try:
        parser.feed(content)
    except Exception:
        return []
    return parser.tables


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


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


def extract_4_columns_from_table(
    table_grid: list[list[str]],
    col_map: dict[str, int],
    param_name: str,
    param_type: str,
    name_col_idx: int = 0,
) -> dict[str, str]:
    """Find param_name in a parsed table and extract the 4 target columns."""
    result: dict[str, str] = {}
    target_row: list[str] | None = None

    for row in table_grid:
        if name_col_idx < len(row):
            cell = row[name_col_idx].strip()
            if cell == param_name or cell.replace("*", "").strip() == param_name:
                target_row = row
                break

    if target_row is None:
        return result

    if "shape" in col_map:
        idx = col_map["shape"]
        if idx < len(target_row):
            val = _clean_cell_value(target_row[idx])
            # "与self一致" etc. is not a real shape — let LLM resolve it.
            if val and not _is_relative_ref(val):
                result["shape"] = val

    if "dtype_desc" in col_map:
        idx = col_map["dtype_desc"]
        if idx < len(target_row):
            val = _clean_cell_value(target_row[idx])
            # "与self一致" etc. is not a real dtype — let LLM resolve it.
            if val and not _is_relative_ref(val):
                result["dtype_desc"] = val

    if "dformat_desc" in col_map:
        idx = col_map["dformat_desc"]
        if idx < len(target_row):
            val = _clean_cell_value(target_row[idx])
            # "与self一致" etc. is not a real format — let LLM resolve it.
            if val and not _is_relative_ref(val):
                result["dformat_desc"] = val

    if "is_support_discontinuous" in col_map:
        idx = col_map["is_support_discontinuous"]
        if idx < len(target_row):
            result["is_support_discontinuous"] = _extract_discontinuous(
                target_row[idx], param_type
            )

    if "param_desc" in col_map:
        idx = col_map["param_desc"]
        if idx < len(target_row):
            result["param_desc"] = _clean_cell_value(target_row[idx])

    if "direction" in col_map:
        idx = col_map["direction"]
        if idx < len(target_row):
            direction = _extract_direction(target_row[idx])
            if direction:
                result["direction"] = direction

    return result


def find_param_name_column(header_row: list[str]) -> int:
    """Detect the parameter-name column index.  Falls back to 0."""
    name_headers = {"参数名", "参数名称", "parameter", "parameter name", "参数"}
    for idx, h in enumerate(header_row):
        norm = _normalise(h)
        for nh in name_headers:
            if _normalise(nh) == norm or _normalise(nh) in norm:
                return idx
    return 0
