# ACLNN CPU Golden Reference Derivation Guide

## Overview

This document provides guidance for deriving CPU golden reference implementations
from ACLNN operator specifications. The CPU golden reference is used to validate
NPU execution results in ATK test cases.

## Derivation Rules

### 1. Tensor Input Mapping

| ACLNN Type | PyTorch CPU Equivalent |
|---|---|
| `const aclTensor*` | `torch.Tensor` (CPU) |
| `aclTensor*` (output) | Pre-allocated `torch.empty()` |
| `const aclScalar*` | Python scalar (`float` / `int`) |
| `int64_t` | Python `int` |
| `float` | Python `float` |
| `bool` | Python `bool` |
| `int64_t*` (array) | Python `list[int]` or `tuple` |

### 2. Common Operator Mappings

#### aclnnAdaLayerNorm → CPU
```python
def cpu_golden_ada_layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return weight * x_norm + bias
```

#### aclnnAdd → CPU
```python
def cpu_golden_add(x1, x2, alpha=1.0):
    return x1 + alpha * x2
```

#### aclnnRelu → CPU
```python
def cpu_golden_relu(x):
    return torch.relu(x)
```

#### aclnnSoftmax → CPU
```python
def cpu_golden_softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)
```

#### aclnnMatmul → CPU
```python
def cpu_golden_matmul(x1, x2):
    return torch.matmul(x1, x2)
```

#### aclnnLayerNorm → CPU
```python
def cpu_golden_layer_norm(x, normalized_shape, weight, bias, eps=1e-5):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
```

#### aclnnGelu → CPU
```python
def cpu_golden_gelu(x):
    return torch.nn.functional.gelu(x)
```

#### aclnnSigmoid → CPU
```python
def cpu_golden_sigmoid(x):
    return torch.sigmoid(x)
```

### 3. Tolerance Settings

| Data Type | Absolute Tolerance | Relative Tolerance |
|---|---|---|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |

### 4. ATK Integration Pattern

Replace the `cpu_golden_reference` placeholder in the ATK executor with the
derived CPU implementation:

```python
def cpu_golden_reference(*args, **kwargs):
    return cpu_golden_<operator>(*args, **kwargs)
```

### 5. Validation

Use `torch.allclose(npu_result, cpu_result, atol=atol, rtol=rtol)` for comparison.
