# aclnnAddmmWeightNz

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

- 接口功能：计算α 乘以mat1与mat2的乘积，再与β和self的乘积求和。相较于原有addmm接口，新接口mat2支持nz格式。

- 计算公式：

$$out = \beta \cdot self + \alpha \cdot (mat1 @ mat2)$$

- 示例：
  - 对于aclnnAddmmWeightNz接口，self的shape是[n,]，mat1的shape是[m, k]，mat2的shape是[k, n]，mat1和mat2的矩阵乘的结果shape是[m, n]，self的shape能broadcast到[m, n]。
  - 对于aclnnAddmmWeightNz接口，self的shape是[1, n]，mat1的shape是[m, k]，mat2的shape是[k, n]，mat1和mat2的矩阵乘的结果shape是[m, n]，self的shape能broadcast到[m, n]。
  - 对于aclnnAddmmWeightNz接口，self的shape是[m, n]，mat1的shape是[m, k]，mat2的shape是[k, n]，mat1和mat2的矩阵乘的结果shape是[m, n]。

## 函数原型

每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用 "aclnnAddmmWeightNzGetWorkspaceSize" 接口获取入参并根据计算流程计算所需workspace大小，再调用 "aclnnAddmmWeightNz"接口执行计算。

```cpp
aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *mat1,
  const aclTensor *mat2,
  const aclScalar *beta,
  const aclScalar *alpha,
  aclTensor       *out,
  int8_t           cubeMathType,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnAddmmWeightNz(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnAddmmWeightNzGetWorkspaceSize

- **参数说明：**

| 参数名 | 输入/输出 | 说明 | 数据类型 | 格式 | 维度 | 是否支持 |
| --- | --- | --- | --- | --- | --- | --- |
| self | 输入 | - shape需要与mat1@mat2满足[broadcast关系](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/common/broadcast%E5%85%B3%E7%B3%BB.md)。<br>- 在mat1不转置的情况下各个维度表示：（m，k）。<br>- 在mat1转置的情况下各个维度表示：（k，m）。 | BFLOAT16、FLOAT16 | ND | 2 | √ |
| mat1 | 输入 | - shape需要与self，mat2满足[broadcast关系](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/common/broadcast%E5%85%B3%E7%B3%BB.md)。<br>- 在mat1不转置的情况下各个维度表示：（m，k）。<br>- 在mat1转置的情况下各个维度表示：（k，m）。 | BFLOAT16、FLOAT16 | ND | 2 | √ |
| mat2 | 输入 | - 当mat2矩阵不转置时，AI处理器亲和数据排布格式各个维度表示：（n1，k1，k0，n0），其中k0 = 16，n0为16。mat1 shape中的k和mat2 shape中的k1需要满足以下关系：ceil（k，k0） = k1，mat2 shape中的n1与out的n满足以下关系：ceil(n, n0) = n1。<br>- 当mat2矩阵转置时，AI处理器亲和数据排布格式各个维度表示：（k1，n1，n0，k0），其中n0 = 16，k0 = 16。mat1 shape中的k和mat2 shape中的k1需要满足以下关系：ceil（k，k0） = k1，mat2 shape中的n1与out的n满足以下关系：ceil(n, n0) = n1。 | BFLOAT16、FLOAT16 | ND | 2 | √ |
| beta(β) | 输入 | - | FLOAT | - | - | - |
| alpha(α) | 输入 | - | FLOAT | - | - | - |
| out | 输出 | - | BFLOAT16、FLOAT16 | ND | 2 | √ |
| cubeMathType | 输入 | - 0：KEEP_DTYPE，保持输入的数据类型进行计算。<br>- 1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。<br>- 2：USE_FP16，支持将输入降精度至FLOAT16计算。<br>- 3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。 | INT8 | - | - | - |
| workspaceSize | 出参 | - | UINT64 | - | - | - |
| executor | 出参 | 用于第二段接口执行。 | - | - | - | - |

- Atlas 训练系列产品、Atlas 推理系列产品：
  - 不支持BFLOAT16数据类型；
  - 当输入数据类型为FLOAT32时不支持cubeMathType=0；
  - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为FLOAT16计算，当输入为其他数据类型时不做处理；
  - 不支持cubeMathType=3。

- Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
  - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理；
  - cubeMathType=2，当输入数据类型为BFLOAT16时不支持该选项；
  - cubeMathType=3，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不支持该选项。

- **返回值：**

aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_nn.md)。

第一段接口完成入参校验，出现如下场景时报错：

| 返回值 | 场景 | 处理建议 |
| --- | --- | --- |
| ACL_ERROR_INVALID_PARAM | self、mat1、mat2、out为空指针 | 检查输入参数是否正确 |
| ACL_ERROR_ILLEGAL_MEMORY_ADDRESS | self、mat1、mat2、out的device地址非法 | 检查输入参数是否正确 |

## aclnnAddmmWeightNz

- **参数说明：**

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| workspace | 输入 | 根据第一段接口计算出的workspace大小，申请device内存。 |
| workspaceSize | 输入 | 根据第一段接口计算出的workspace大小。 |
| executor | 输入 | 第一段接口的输出，用于第二段接口执行。 |
| stream | 输入 | aclrtStream类型，用于指定执行的异步通道。 |

- **返回值：**

aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_nn.md)。

## 约束说明

- 确定性说明：
  - Atlas 训练系列产品、Atlas 推理系列产品：aclnnAddmmWeightNz默认确定性实现。

- 计算一致性说明：
  - Atlas 训练系列产品、Atlas 推理系列产品：
    - 当开启强一致性计算功能时，计算结果时确定的，多次执行将产生相同的输出。此外，计算结果与数据的位置无关。
    - aclnnAddmmWeightNz默认非一致性实现，支持通过aclrtCtxSetSysParamOpt开启一致性。
    - 例如，在进行矩阵乘时，不同基本块的累加顺序可能不同，这可能会导致相同数据在不同行的计算结果出现细微差异。然而，在开启强一致性计算的情况下，即使在不同的行中，只要输入相同，计算结果也将相同。

- 不支持mat1与mat2两个输入中一个输入为BFLOAT16，另一个输入为FLOAT或者FLOAT16的数据类型推导。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/context/common/%E7%BC%96%E8%AF%91%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%A0%B7%E4%BE%8B_nn.md)。

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_addmm.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"
#include "aclnnop/aclnn_cast.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

// 将FP16的uint16_t表示转换为float表示
float Fp16ToFloat(uint16_t h) {
  int s = (h >> 15) & 0x1;              // sign
  int e = (h >> 10) & 0x1F;             // exponent
  int f =  h        & 0x3FF;            // fraction
  if (e == 0) {
    // Zero or Denormal
    if (f == 0) {
      return s ? -0.0f : 0.0f;
    }
    // Denormals
    float sig = f / 1024.0f;
    float result = sig * pow(2, -24);
    return s ? -result : result;
  } else if (e == 31) {
      // Infinity or NaN
      return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
  }
  // Normalized
  float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
  return s ? -result : result;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorWeight(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                          aclDataType dataType, aclTensor** tensor) {
  auto size = static_cast<uint64_t>(GetShapeSize(shape));

  const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
  auto ret = aclnnCalculateMatmulWeightSize(mat2Size, &size);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSize failed. ERROR: %d\n", ret); return ret);
  size *= sizeof(T);

  // 调用aclrtMalloc申请device侧内存
  ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  std::vector<int64_t> storageShape;
  storageShape.push_back(GetShapeSize(shape));

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            storageShape.data(), storageShape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {16};
  std::vector<int64_t> mat1Shape = {16, 32};
  std::vector<int64_t> mat2Shape = {32, 16};
  std::vector<int64_t> outShape = {16, 16};
  void* selfDeviceAddr = nullptr;
  void* mat1DeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat1 = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;
  aclScalar* alpha = nullptr;
  aclScalar* beta = nullptr;

  std::vector<uint16_t> selfHostData(16, 0x3C00); // float16_t 用0x3C00表示int_16的1
  std::vector<uint16_t> mat1HostData(512, 0x3C00); // float16_t 用0x3C00表示int_16的1
  std::vector<uint16_t> mat2HostData(512, 0x3C00); // float16_t 用0x3C00表示int_16的1
  std::vector<uint16_t> outHostData(256, 0);
  float alphaValue = 1.0f;
  float betaValue = 1.0f;

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mat1 aclTensor
  ret = CreateAclTensor(mat1HostData, mat1Shape, &mat1DeviceAddr, aclDataType::ACL_FLOAT16, &mat1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mat2 aclTensor
  ret = CreateAclTensorWeight(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建alpha aclScalar
  alpha = aclCreateScalar(&alphaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // 创建beta aclScalar
  beta = aclCreateScalar(&betaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);


  // 3. 调用CANN算子库API，需要修改为具体的Api名
  int8_t cubeMathType = 1;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用TransWeight
  ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnTransMatmulWeight第二段接口
  ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

  // 调用aclnnAddmmWeightNz第一段接口
  uint64_t workspaceSizeMm = 0;
  ret = aclnnAddmmWeightNzGetWorkspaceSize(self, mat1, mat2, beta, alpha, out, cubeMathType, &workspaceSizeMm, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmmWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddrMm = nullptr;
  if (workspaceSizeMm > 0) {
    ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnAddmmWeightNz第二段接口
  ret = aclnnAddmmWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmmWeightNz failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成float表示的fp16
  for (int64_t i = 0; i < size; i++) {
    float fp16Float = Fp16ToFloat(resultData[i]);
    LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(mat1);
  aclDestroyTensor(mat2);
  aclDestroyScalar(beta);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(mat1DeviceAddr);
  aclrtFree(mat2DeviceAddr);
  aclrtFree(outDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (workspaceSizeMm > 0) {
    aclrtFree(workspaceAddrMm);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
