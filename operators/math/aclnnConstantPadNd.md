# aclnnConstantPadNd

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/pad_v3)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入的张量self，以pad参数为基准进行数据填充，填充值为value。

- 计算公式：

  - out tensor的shape推导公式：

    $$
    \begin{align}
    假设输入&self的shape为：\\
    [dim0_{in},& dim1_{in}, dim2_{in}, dim3_{in}] \\
    假设pad  = &\lbrace dim3_{begin},dim3_{end},\\
    & dim2_{begin},dim2_{end},\\
    & dim1_{begin},dim1_{end},\\
    & dim0_{begin},dim0_{end} \rbrace
    \end{align}
    $$
    
    $$
    \begin{aligned}
    &则输出out的shape为：
    \\ &[dim0_{out}, dim1_{out}, dim2_{out}, dim3_{out}] =
    &[dim0_{begin}+dim0_{in}+dim0_{end},
    \\&&dim1_{begin}+dim1_{in}+dim1_{end},
    \\&&dim2_{begin}+dim2_{in}+dim2_{end},
    \\&&dim3_{begin}+dim3_{in}+dim3_{end}]
    \end{aligned}
    $$

  - 例子1：  
    (pad数组长度等于self维度的两倍的情况)

    $$
    \begin{aligned}
    selfShape &= [1, 1, 1, 1, 1]\\
    pad &= \lbrace 0, 1, 2, 3, 4, 5, 6, 7, 8, 9\rbrace \\
    outputShape &= [8+1+9, 6+1+7, 4+1+5, 2+1+3, 0+1+1]\\
    &= [18,14,10,6,2]
    \end{aligned}
    $$

  - 例子2：  
    (pad数组长度小于self维度的两倍的情况)
    
    $$
    \begin{aligned}
    selfShape &= [1, 1, 1, 1, 1]\\
    pad &= \lbrace 0, 1, 2, 3, 4, 5\rbrace \\
    outputShape &= [0+1+0, 0+1+0, 4+1+5, 2+1+3, 0+1+1]\\
    &= [1,1,10,6,2]
    \end{aligned}
    $$

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnConstantPadNdGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConstantPadNd”接口执行计算。

  - `aclnnStatus aclnnConstantPadNdGetWorkspaceSize(const aclTensor* self, const aclIntArray* pad, const aclScalar* value, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnConstantPadNd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnConstantPadNdGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：待填充的原输入数据，Device侧的aclTensor。shape支持0-8维，value与self的数据类型满足数据类型推导规则（参见[互推导关系](../common/互推导关系.md)），支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128。
  - pad(aclIntArray*, 计算输入)：输入中各轴需要填充的维度，host侧的aclIntArray。数组长度必须为偶数且不能超过self维度的两倍。

  - value(aclScalar*, 计算输入)：填充部分的填充值，host侧的aclScalar。value与self的数据类型满足数据类型推导规则（参见[互推导关系](../common/互推导关系.md)）。

  - out(aclTensor*, 计算输出)：填充后的输出结果，Device侧的aclTensor。shape需要满足示例中的推导规则，数据类型需要与self一致，支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128。
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001 (ACLNN_ERR_PARAM_NULLPTR)： 
                                      1. 传入的self、pad、value或out是空指针。
返回161002 (ACLNN_ERR_PARAM_INVALID)： 
                                      1. self、value或out的数据类型不在支持的范围之内。
                                      2. self与out的数据类型不一致。
                                      3. self与value的数据类型不满足数据类型推导规则。
                                      4. self的shape和pad的输入推导出的shape与out的shape不一致。
                                      5. pad中元素不为偶数或超过了self维度的两倍。
                                      6. self或out的维度大于8。
                                      7. pad中每个值都不能让out的shape小于0，如果pad中存在正数，则out的shape中不能有0。
                                      8. 当self的数据格式不为ND，out的数据格式与self的数据格式不一致。
```

## aclnnConstantPadNd。

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnConstantPadNdGetWorkspaceSize获取。

  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream, 入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnConstantPadNd默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_constant_pad_nd.h"

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
  std::vector<int64_t> selfShape = {3, 3};
  std::vector<int64_t> outShape = {5, 5};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* pad = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> outHostData(25, 0);
  float valueValue = 0.0f;
  std::vector<int64_t> padData = {1,1,1,1};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建pad数组
  pad = aclCreateIntArray(padData.data(), 4);
  CHECK_RET(pad != nullptr, return ret);
  // 创建value aclScalar
  value = aclCreateScalar(&valueValue, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnConstantPadNd第一段接口
  ret = aclnnConstantPadNdGetWorkspaceSize(self, pad, value, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConstantPadNdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnConstantPadNd第二段接口
  ret = aclnnConstantPadNd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConstantPadNd failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyIntArray(pad);
  aclDestroyScalar(value);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
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
