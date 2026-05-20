# aclnnObfuscationCalculate


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

 - 算子功能：将张量x和配置参数（如param、cmd）发送至PMCC混淆引擎。引擎的CA模块调用TA模块，进行张量混淆处理，最终返回shape与x一致的混淆后的张量y。
 
 - 背景：PMCC（Privacy&Model Confidential Computing）模型混淆特性利用CPU核中的TrustZone可信执行环境隔离存储混淆因子、派生混淆掩码、执行动态掩码添加。PMCC基于NPU TrustZone构建了模型混淆引擎CA（普通OS中的Client Application）与模型混淆引擎TA（TEE OS中的Trusted Application）。为了使模型在推理执行过程中能够访问模型混淆引擎TA，通过AICPU算子机制及NPU卡内localhost socket进行中转。

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用 “aclnnObfuscationCalculateGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnObfuscationCalculate”接口执行计算。

- `aclnnStatus aclnnObfuscationCalculateGetWorkspaceSize(int32_t fd, const aclTensor* x, int32_t param, int32_t cmd, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnObfuscationCalculate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnObfuscationCalculateGetWorkspaceSize

* **参数说明**：

  * fd（int32_t ，计算输入）：socket连接符，数据类型为INT32，填写aclnnObfuscationSetup在资源初始化时输出中的fd[0]。
  * x（aclTensor* ，计算输入）：待混淆处理的张量，Shape为(*,*,...,hiddenSize)，Shape最后一维的size是aclnnObfuscationSetup在资源初始化时配置的hiddenSize，[数据格式](../common/数据格式.md)要求为ND。不支持[非连续的Tensor](../common/非连续的Tensor.md)，不支持空Tensor。
      * <term>Atlas 推理系列产品</term>：Tensor数据类型支持FLOAT、FLOAT16、INT8。
      * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：Tensor数据类型支持FLOAT、FLOAT16、INT8、BFLOAT16。
  * param（int32_t，计算输入）：预留的参数字段，数据类型为INT32，当前版本仅支持0。
  * cmd（int32_t，计算输入）：混淆算子指令编号，当前版本仅支持1。
  * y（aclTensor*，计算输出）：混淆处理后的张量，数据类型及Shape与x相同，[数据格式](../common/数据格式.md)要求为ND。不支持[非连续的Tensor](../common/非连续的Tensor.md)，不支持空Tensor。
  * workspaceSize（uint64_t*，出参）：返回用户需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

* **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_nn.md)。

  ```
  第一段接口会完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x或y是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x的数据类型和数据格式不在支持的范围之内。
                                       2. x和y的数据类型不一致。
                                       3. x和y的形状不一致。
  ```

## aclnnObfuscationCalculate
* **参数说明：**

  * workspace（void*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnObfuscationCalculateGetWorkspaceSize获取。
  * executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的Stream。
* **返回值：**
  返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_nn.md)。

## 约束说明
- 确定性计算：
  - aclnnObfuscationCalculate默认确定性实现。

- 该接口与[aclnnObfuscationSetup](./aclnnObfuscationSetup.md)配套使用，完成PMCC模型混淆功能，使用方式如下：
  - 首先调用aclnnObfuscationSetup进行资源初始化，可重复调用，以最后一次初始化为准
  - 再多次调用aclnnObfuscationCalculate进行张量混淆处理
  - 最后调用aclnnObfuscationSetup进行资源释放，只能调用一次；也可不显式进行资源释放，而是通过终止程序进程的方式达到资源释放的目的

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_obfuscation_setup.h"
#include "aclnnop/aclnn_obfuscation_calculate.h"

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
  std::vector<int64_t> fdShape = {1};
  void* fdDeviceAddr = nullptr;

  int32_t fdToClose = 0;
  int32_t dataType = 0;
  int32_t hiddenSize = 4;
  int32_t tpRank = 0;
  int32_t modelObfSeedId = 123456;
  int32_t dataObfSeedId = 654321;
  int32_t cmd = 1;
  int32_t threadNum = 4;
  aclTensor* fd = nullptr;
  std::vector<float> fdHostData = {-1};

  //创建fd aclTensor 
  ret = CreateAclTensor(fdHostData, fdShape, &fdDeviceAddr, aclDataType::ACL_INT32, &fd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnObfuscationSetup第一段接口
  ret = aclnnObfuscationSetupGetWorkspaceSize(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, fd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnObfuscationSetup第二段接口
  ret = aclnnObfuscationSetup(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetup failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto fdSize = GetShapeSize(fdShape);
  std::vector<int32_t> fdData(fdSize, 0);
  ret = aclrtMemcpy(fdData.data(), fdData.size() * sizeof(fdData[0]), fdDeviceAddr, fdSize * sizeof(int32_t),ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR : %d\n", ret); return ret);
  for (int64_t i = 0; i < fdSize; i++) {
      LOG_PRINT("fdData[%ld] is : %d\n", i, fdData[i]);
  }

  // 6. 构造输入与输出, 需要根据API的接口定义构造
  std::vector<int64_t> xShape = {2, 4};
  std::vector<int64_t> yShape = {2, 4};
  void *xDeviceAddr = nullptr;
  void *yDeviceAddr = nullptr;

  int32_t fdInput = fdData[0];
  int32_t param = 4;
  aclTensor *x = nullptr;
  int32_t cmd2 = 1;
  aclTensor *y = nullptr;
  std::vector<float> xHostData = {0.86, 0.79, 0.43, 0.37, 0.51, 0.89, 0.34, 0.49};
  std::vector<float> yHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  // 创建 x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建 y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 7. 调用CANN算子库API, 需要修改为具体的API
  uint64_t workspaceSize2 = 0;
  aclOpExecutor *executor2;

  // 调用aclnnObfuscationCalculate第一段接口
  ret = aclnnObfuscationCalculateGetWorkspaceSize(fdInput, x, param, cmd2, y, &workspaceSize2, &executor2);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationCalculateGetWorkspaceSize failed. ERROR : %d\n",ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr2 = nullptr;
  if (workspaceSize2 > 0) {
      ret = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnObfuscationCalculate第二段接口
  ret = aclnnObfuscationCalculate(workspaceAddr2, workspaceSize2, executor2, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationCalculate failed. ERROR : %d\n", ret); return ret);
  // 8. 固定写法，同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR : %d\n", ret); return ret);

  // 9. 获取输出的值，y表示经过混淆处理的数据，将device侧的内存上的结果拷贝至host侧, 需要根据具体API的接口定义修改
  // y
  auto ySize = GetShapeSize(yShape);
  std::vector<float> yData(ySize, 0);
  ret = aclrtMemcpy(yData.data(), yData.size() * sizeof(yData[0]), yDeviceAddr, ySize * sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR : %d\n", ret); return ret);
  for (int64_t i = 0; i < ySize; i++) {
      LOG_PRINT("yData[%ld] is : %f\n", i, yData[i]);
  }

   // 10. 构造输入与输出，需要根据API的接口自定义构造
  fdToClose = fdInput;
  dataType = 0;
  hiddenSize = 0;
  tpRank = 0;
  modelObfSeedId = 0;
  dataObfSeedId = 0;
  cmd = 16;
  threadNum = 4;

  // 11. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize3 = 0;
  aclOpExecutor* executor3;

  // 调用aclnnObfuscationSetup第一段接口
  ret = aclnnObfuscationSetupGetWorkspaceSize(fdToClose, dataType, hiddenSize, tpRank, modelObfSeedId,dataObfSeedId, cmd, threadNum, fd, &workspaceSize3, &executor3);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetupGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr3 = nullptr;
  if (workspaceSize3 > 0) {
      ret = aclrtMalloc(&workspaceAddr3, workspaceSize3, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnObfuscationSetup第二段接口
  ret = aclnnObfuscationSetup(workspaceAddr3, workspaceSize3, executor3, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnObfuscationSetup failed. ERROR: %d\n", ret); return ret);

  // 12. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 13. 释放ObfuscationCalculate接口涉及的aclTensor和aclScalar, 需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(y);
  aclDestroyTensor(fd);

  // 14. 释放ObfuscationCalculate接口涉及的device资源, 需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(fdDeviceAddr);

  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  if (workspaceSize2 > 0) {
      aclrtFree(workspaceAddr2);
  }  
  if (workspaceSize3 > 0) {
      aclrtFree(workspaceAddr3);
  }
  // 15. 释放ObfuscationSetup接口涉及的device资源
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
