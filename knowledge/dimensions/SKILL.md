---
name: dimensions-generation
description: >
  Convert CANN operator parameter shape descriptions into structured
  dimensions arrays. Supports rank format [min,max], per-dimension format
  [[min,max],...], numeric/symbolic dimensions, HTML list variants, and edge cases.
license: MIT
---

# Dimensions Generation Rules

## 1. Output Formats

| Format | Meaning | Example |
|--------|---------|---------|
| `[v1, v2, ...]` (enum) | supported dimension counts | `[2,3,4,5,6]` = 2D to 6D |
| `[v1, v2, ...]` (enum) | discrete dimension counts | `[0,3,4]` = 0D or 3D or 4D |
| `[N]` (single) | exactly N dimensions | `[4]` = exactly 4D |
| `[]` (empty) | scalar / unknown / cross-param ref | |

## 2. Parsing Rules

### Rank patterns
| Shape text | Output | Rule |
|------------|--------|------|
| `"0、3、4"` / `"0,3,4"` | `[0, 3, 4]` | Discrete enumeration (NEW) |
| `"2-6"` / `"2~6"` | `[2,3,4,5,6]` | Range -> expand to enumeration |
| `"2D"` / `"3-D"` | `[2]` / `[3]` | Exact rank (D suffix) |
| `"2维~6维"` | `[2,3,4,5,6]` | Chinese range -> expand |
| `"1维"` / `"3维"` | `[1]` / `[3]` | Chinese rank exact (维 suffix) |
| `"1维，最大长度256"` | `[1]` | Chinese rank + size info (only rank matters) |

**重要：维数与长度的区分**
- "N维" 描述的是 tensor 的**维度数**（rank），应输出 `[N]`
- "最大长度M" / "最大长度为M" 描述的是某一维的**大小限制**，不属于 dimensions，
  应由 constraints_in_parameters 中的 `self_shape_dim_range` 约束表达
- 例如 "1维，最大长度256" → dimensions `[1]`，长度限制 256 在 constraints 中
- **不要**将 "1维，最大长度256" 解析为 `[[1, 256]]`（旧 per-dim 格式），这是错误的

### Tuple / bracket patterns
| Shape text | Output | Rule |
|------------|--------|------|
| `"(N,C,H,W)"` | `[4]` | Symbolic tuple → count comma slots |
| `"(H*rankSize, N)"` | `[2]` | Compound expr → count slots |
| `"[2, 3, 4]"` | `[3]` | Numeric bracket → count slots as rank |
| `"[8]"` | `[1]` | Single numeric dim |
| `"[K1, N1]"` | `[2]` | Symbolic bracket → count slots as rank |

### Scalar / empty / cross-reference → `[]`
| Shape text | Output |
|------------|--------|
| `"标量"` / `"0-D"` / `"scalar"` | `[]` |
| `""` / `"-"` / `"N/A"` | `[]` |
| `"与输入相同"` / `"同输入"` / `"same as input"` | `[]` |
| `"与weight1一致"` / `"与xxx的转置一致"` | `[]` (cross-param ref, deferred) |

### HTML list shapes (quantization params)
When shape contains `<ul><li>` with multiple bracket variants like
`[E, N1]/[N1]` (per-channel / per-group / per-tensor × with/without experts):
1. Extract every `[...]` bracket group from the raw text
2. Count comma-separated slots in each → that is the rank
3. Return `sorted(set(counts))` — the distinct ranks that actually appear

Example:
`<ul><li>per-channel...[E, N1]/[N1]</li><li>per-group...[E, G, N1]/[G, N1]</li></ul>`
→ brackets: [E,N1]=2, [N1]=1, [E,G,N1]=3, [G,N1]=2 → counts={1,2,3} → `[1, 2, 3]`

**注意：不要用 `_expand_range(min, max)`**，各变体的 rank 不一定连续。
例如 counts={1,3} 时应输出 `[1, 3]` 而非 `[1, 2, 3]`。

## 3. Platform Awareness
Shape values come as JSON: `{"*": value}` or `{"platform": value}`.
Each platform's shape is converted independently.

## 4. Validation
- All values must be integers in `[0, 8]`
- List must be sorted ascending and deduplicated
- `[]` is always valid
