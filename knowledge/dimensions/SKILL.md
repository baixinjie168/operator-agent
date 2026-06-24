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
| `[min, max]` (rank) | dimension count range | `[4, 4]` = exactly 4D |
| `[[min,max], ...]` (per-dim) | size of each dimension | `[[2,2],[3,3]]` = 2D, sizes 2,3 |
| `[]` (empty) | scalar / unknown / cross-param ref | |

## 2. Parsing Rules

### Rank patterns
| Shape text | Output | Rule |
|------------|--------|------|
| `"0-8"` / `"2~6"` | `[0, 8]` / `[2, 6]` | Rank range |
| `"2D"` / `"3-D"` | `[2, 2]` / `[3, 3]` | Rank exact (D suffix) |
| `"1D~8D"` | `[1, 8]` | Rank range with D suffix |
| `"2维~8维"` | `[2, 8]` | Chinese dimension range |

### Tuple / bracket patterns
| Shape text | Output | Rule |
|------------|--------|------|
| `"(N,C,H,W)"` | `[4, 4]` | Symbolic tuple → count comma slots |
| `"(H*rankSize, N)"` | `[2, 2]` | Compound expr → count slots |
| `"[2, 3, 4]"` | `[[2,2],[3,3],[4,4]]` | Pure numeric → per-dimension |
| `"[8]"` | `[[8,8]]` | Single numeric dim |
| `"[0-100, 0-200]"` | `[[0,100],[0,200]]` | Per-dim with ranges |

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
3. Return `[min_rank, max_rank]` across ALL variants

Example:
`<ul><li>per-channel...[E, N1]/[N1]</li><li>per-group...[E, G, N1]/[G, N1]</li></ul>`
→ brackets: [E,N1]=2, [N1]=1, [E,G,N1]=3, [G,N1]=2 → `[1, 3]`

## 3. Platform Awareness
Shape values come as JSON: `{"*": value}` or `{"platform": value}`.
Each platform's shape is converted independently.

## 4. Validation
- rank: `0 <= min <= max <= 10`
- per-dim: each `min <= max` (or null), max 10 dimensions
- `[]` is always valid
