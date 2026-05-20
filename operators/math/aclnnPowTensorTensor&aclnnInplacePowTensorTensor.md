# aclnnPowTensorTensor&aclnnInplacePowTensorTensor

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/pow)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：exponent每个元素作为input对应元素的幂完成计算。

- 计算公式：

  $$
  out_i = x_i^{exponent_i}
  $$
  
- 算子约束：INT32整型计算在如下范围以外的场景，会出现超时；

  | shape  | exponent_value|
  |----|----|
  |<=100000（十万） |-200000000~200000000(两亿)|
  |<=1000000（百万） |-20000000~20000000(两千万)|
  |<=10000000（千万） |-2000000~2000000(两百万)|
  |<=100000000（亿） |-200000~200000(二十万)|
  |<=1000000000（十亿） |-20000~20000(两万)|

## 函数原型

- aclnnPowTensorTensor和aclnnInplacePowTensorTensor实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnPowTensorTensor：需新建一个输出张量对象存储计算结果。
  - aclnnInplacePowTensorTensor：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnPowTensorTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPowTensorTensor”接口执行计算。

  - `aclnnStatus aclnnPowTensorTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* exponent, aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnPowTensorTensor(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplacePowTensorTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* exponent, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplacePowTensorTensor(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, aclrtStream stream)`

## aclnnPowTensorTensorGetWorkspaceSize

- **参数说明**：

  - self(aclTensor*，计算输入)：Device侧的aclTensor。且数据类型需要与exponent满足数据类型推导规则（参见[互推导关系](../common/互推导关系.md)），shape支持0-8维，self和exponent数据类型不支持同时为BOOL，shape需要与exponent满足[broadcast关系](../common/broadcast关系.md)。支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT16、BOOL、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、BFLOAT16。
    * <term>Atlas 训练系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT16、BOOL、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128。

  - exponent(aclTensor*，计算输入)：Device侧的aclTensor。数据类型需要与self满足数据类型推导规则（参见[互推导关系](../common/互推导关系.md)），self和exponent数据类型不支持同时为BOOL，shape需要与self满足[broadcast关系](../common/broadcast关系.md)。支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT16、BOOL、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、BFLOAT16。
    * <term>Atlas 训练系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT16、BOOL、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128。

  - out(aclTensor*，计算输出)：Device侧的aclTensor。数据类型需要是self与exponent推导之后可转换的数据类型，shape需要是self与exponent broadcast之后的shape。支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT16、BOOL、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、BFLOAT16。
    * <term>Atlas 训练系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT16、BOOL、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128。

  - workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor **，出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、exponent、out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self和exponent的数据类型不在支持的范围之内。
                                        2. self和exponent无法做数据类型推导。
                                        3. 推导出的数据类型无法转换为指定输出out的类型。
                                        4. self和exponent的shape无法做broadcast。
                                        5. self和exponent同时为bool数据类型。
  ```

## aclnnPowTensorTensor

- **参数说明**：

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnPowTensorTensorGetWorkspaceSize获取。

  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream，入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

## aclnnInplacePowTensorTensorGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入|计算输出)：输入输出tensor，即公式中的输入x与输出out。数据类型与exponent的数据类型需满足数据类型推导规则（参见[互推导关系](../common/互推导关系.md)），且数据类型是self与exponent推导之后可转换的数据类型（参见[互转换关系](../common/互转换关系.md)），shape需要与exponent满足[broadcast关系](../common/broadcast关系.md)。支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、INT16、BFLOAT16。
    * <term>Atlas 训练系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT8、UINT8、COMPLEX64、COMPLEX128、INT16。
  - exponent(aclTensor*, 计算输入)：公式中的输入`exponent`。数据类型需要与self满足数据类型推导规则（参见[互推导关系](../common/互推导关系.md)），shape需要与self满足[broadcast关系](../common/broadcast关系.md)。支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、BOOL、INT8、UINT8、COMPLEX64、COMPLEX128、INT16、BFLOAT16。
    * <term>Atlas 训练系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、BOOL、INT8、UINT8、COMPLEX64、COMPLEX128、INT16。
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、exponent空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self和exponent的数据类型不在支持的范围之内。
                                        2. self和exponent的shape大于8维。
                                        3. self和exponent无法满足数据类型推导规则。
                                        4. self和other的shape无法做broadcast。
                                        5. self和exponent进行broadcast后的shape不等于self的shape。
                                        6. self和exponent同时为bool数据类型。
  ```

## aclnnInplacePowTensorTensor

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplacePowTensorTensorGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnPowTensorTensor&aclnnInplacePowTensorTensor默认确定性实现。

<term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：该场景下，如果计算结果取值超过了设定的数据类型取值范围，则会以该数据类型的边界值作为结果返回。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例.md)。
**aclnnPowTensorTensor示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

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

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> expShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建exp aclTensor
  ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnPowTensorTensor第一段接口
  ret = aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnPowTensorTensor第二段接口
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(exp);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(expDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
**aclnnInplacePowTensorTensor示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

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

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfRefShape = {4, 2};
  std::vector<int64_t> expShape = {4, 2};
  void* selfRefDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* exp = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  // 创建selfRef aclTensor
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建exp aclTensor
  ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplacePowTensorTensor第一段接口
  ret = aclnnInplacePowTensorTensorGetWorkspaceSize(selfRef, exp, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplacePowTensorTensor第二段接口
  ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),         selfRefDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(selfRef);
  aclDestroyTensor(exp);

  // 7. 释放device资源
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(expDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
