---
name: implicit-params-validation
description: >
  Validate candidate named identifiers regex-extracted from CANN operator
  document shape descriptions.  Distinguish true named dimension variables
  from concept terms, operation names, and data-type names.
license: MIT
---

# Implicit Parameter Validation Rules

## 1. Named dimension variables (confirm)

A candidate is a **named dimension variable** when it represents a symbolic
size value in a shape tuple, not a concept or operation name.

### 1.1 Standard dimension variables

| Identifier | Context | Verdict |
|------------|---------|---------|
| N, C, H, W | shape (N, C, H, W) | dimension variable |
| BS, B | shape (BS, H) | dimension variable |
| batchSize, numHeads | shape (batchSize, numHeads) | dimension variable |
| k0, n0, m0 | shape (k0, n0) | dimension variable (unless assigned -> constant) |
| dim, rank, seqLen | shape (dim, rank) | dimension variable |

### 1.2 Compound expression variables

| Expression | Variable | Verdict |
|------------|----------|---------|
| H*rankSize | H | dimension variable |
| H*rankSize | rankSize | external constant (only in compound expr) |
| BS/rankSize | BS | dimension variable |
| A*B (both appear standalone) | A, B | both dimension variables |

## 2. Concept terms (remove)

### 2.1 Dimension concept names

When "X维度" appears in text, X describes the *meaning* of the dimension,
not a variable name.  If X itself is an operation/algorithm/concept name,
it must be removed.

| Text | Identifier | Verdict | Reason |
|------|-----------|---------|--------|
| mat2的Reduce维度需要与self的Reduce维度大小相等 | Reduce | remove | Reduce is a reduce-operation concept name |
| GEMV维度 | GEMV | remove | GEMV is a matrix-vector multiply operation name |
| Attention维度 | Attention | remove | Attention is an attention mechanism name |
| Conv维度 | Conv | remove | Conv is a convolution operation name |

### 2.2 Operation / algorithm names

| Identifier | Verdict | Reason |
|-----------|---------|--------|
| Softmax, ReLU, Sigmoid, GELU | remove | activation function names |
| LayerNorm, BatchNorm | remove | normalization operation names |
| Conv, Conv2D, Conv3D | remove | convolution operation names |
| Matmul, BMM, MM | remove | matrix multiply operation names |
| Transpose, Reshape, Permute | remove | tensor operation names |

### 2.3 Generic descriptor words

| Identifier | Verdict |
|-----------|---------|
| shape, dtype, format, type | remove |
| input, output, tensor, optional | remove |
| true, false, none, null | remove |
| float, double, int, char, void | remove |

## 3. Constants (reclassify -> constant)

When the text contains an explicit assignment, reclassify the dimension
variable as a constant:

| Text pattern | Identifier | Classification | constant_value |
|-------------|-----------|-----------------|----------------|
| "其中k0 = 16" | k0 | constant | 16 |
| "n0为16" | n0 | constant | 16 |
| "k0等于16" | k0 | constant | 16 |
| "k0 is 16" | k0 | constant | 16 |
| "其中 G = 128" | G | constant | 128 |

## 4. External constants (reclassify -> external_constant)

An identifier is an external constant when it **only** appears in compound
expressions and never as a standalone dimension slot.  It typically depends
on platform configuration.

| Identifier | Occurrence | Verdict | Reason |
|-----------|-----------|---------|--------|
| rankSize | H*rankSize | external_constant | platform-dependent (NPU card count) |
| worldSize | BS/worldSize | external_constant | distributed-training related |
| padSize | N+padSize | external_constant | may be a constant or external parameter |

## 5. Missed parameters (additions)

The regex may miss variables that appear only in constraint description text
and not in any shape tuple.  If you discover such a variable in the
section_text, add it to the additions list.

Example: text says "rankSize 的取值依赖于 NPU 卡数" but rankSize does not
appear in any (xxx) shape tuple.  The Agent should add it to additions.
