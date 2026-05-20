# aclnnCeil&aclnnInplaceCeil

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/ceil)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：返回输入tensor中每个元素向上取整的结果

- 计算公式：

$$
out_i =⌈self_i⌉
$$

## 函数原型

- aclnnCeil和aclnnInplaceCeil实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnCeil：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceCeil：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnCeilGetWorkspaceSize”或者“aclnnInplaceCeilGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCeil”或者“aclnnInplaceCeil”接口执行计算。
  - `aclnnStatus aclnnCeilGetWorkspaceSize(const aclTensor *self,  aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnCeil(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceCeilGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceCeil(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnCeilGetWorkspaceSize

- **参数说明：**
  
  * self(aclTensor*,计算输入)：公式中的self，Device侧的aclTensor，支持[非连续的Tensor](../common/非连续的Tensor.md)，shape维度不高于8维，shape和数据类型必须和out一样，[数据格式](../common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、BFLOAT16。   
  * out(aclTensor \*，计算输出)：公式中的out，Device侧的aclTensor，shape维度不高于8维，shape和数据类型必须和self一样，[数据格式](../common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、BFLOAT16。  
  * workspaceSize(uint64_t \*, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \**, 出参)：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)

```
第一段接口完成入参校验，出现以下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self或out是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. self的数据类型和数据格式不在支持的范围之内。
                                 2. self和out数据类型不一致。
                                 3. self和out的shape不一致。
                                 4. self维度大于8
```

## aclnnCeil

- **参数说明：**
  
  * workspace(void \*，入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnCeilGetWorkspaceSize获取。
  * executor(aclOpExecutor \*，入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream,入参)：指定执行任务的Stream。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

## aclnnInplaceCeilGetWorkspaceSize

- **参数说明：**
  
  * selfRef(aclTensor \*，计算输入|计算输出)：公式中的self，Device侧的aclTensor，支持[非连续的Tensor](../common/非连续的Tensor.md)，shape维度不高于8维，[数据格式](../common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE、BFLOAT16。
  * workspaceSize(uint64_t \*, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \**, 出参)：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的selfRef是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. selfRef的数据类型和数据格式不在支持的范围之内。
                                 2. selfRef维度大于8
```

## aclnnInplaceCeil

- **参数说明：**
  
  * workspace(void \*，入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnCeilGetWorkspaceSize获取。
  * executor(aclOpExecutor \*，入参)aclOpExecutor，包含了算子计算流程。
  * stream(aclrtStream,入参)：指定执行任务的Stream。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnCeil&aclnnInplaceCeil默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ceil.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& inputShape, std::vector<int64_t>& outShape, void** selfDeviceAddr, void** outDeviceAddr,
    aclTensor** self, aclTensor** out)
{
  std::vector<double> selfHostData = {0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
  std::vector<double> outHostData(8, 0);

  // 创建 self aclTensor
  auto ret = CreateAclTensor(selfHostData, inputShape, selfDeviceAddr, aclDataType::ACL_DOUBLE, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 out aclTensor
  ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_DOUBLE, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* out, void** workspaceAddrOut, uint64_t& workspaceSize, void* selfDeviceAddr,
    void* outDeviceAddr, std::vector<int64_t>& outShape, aclrtStream stream)
{
  aclOpExecutor* executor;
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);

  // aclnnCeil 接口调用示例
  LOG_PRINT("test aclnnCeil\n");

  // 调用 aclnnCeil 第一段接口
  auto ret = aclnnCeilGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCeilGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据 workspaceSize 申请 device 内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // 调用 aclnnCeil 第二段接口
  ret = aclnnCeil(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCeil failed. ERROR: %d\n", ret); return ret);

  // 同步
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷贝 out 结果
  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  // aclnnInplaceCeil 调用示例
  LOG_PRINT("\ntest aclnnInplaceCeil\n");

  // 调用 aclnnInplaceCeil 第一段接口
  ret = aclnnInplaceCeilGetWorkspaceSize(self, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCeilGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据 workspaceSize 再次申请 device 内存（与原逻辑一致）
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // 调用 aclnnInplaceCeil 第二段接口
  ret = aclnnInplaceCeil(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCeil failed. ERROR: %d\n", ret); return ret);

  // 同步
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷贝 self 结果
  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  // 1. device/stream 初始化
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InitAcl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  std::vector<int64_t> inputShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  ret = CreateInputs(inputShape, outShape, &selfDeviceAddr, &outDeviceAddr, &self, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用 CANN 算子 API（ceil + inplace ceil）
  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(self, out, &workspaceAddr, workspaceSize, selfDeviceAddr, outDeviceAddr, outShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 6. 释放 aclTensor
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. 释放 device 资源
  aclrtFree(selfDeviceAddr);
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
