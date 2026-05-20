# aclnnObfuscationSetup

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |     ×      |


## 功能说明

 - 算子功能：完成PMCC模型混淆引擎的资源初始化和释放。

   - 资源初始化：与PMCC混淆引擎CA建立socket连接、对CA、TA进行初始化，并返回socket连接符。
   - 资源释放：与PMCC混淆引擎CA断开socket连接。

 - 背景：PMCC（Privacy&Model Confidential Computing）模型混淆特性利用CPU核中的TrustZone可信执行环境隔离存储混淆因子、派生混淆掩码、执行动态掩码添加。PMCC基于NPU TrustZone构建了模型混淆引擎CA（普通OS中的Client Application）与模型混淆引擎TA（TEE OS中的Trusted Application）。为了使模型在推理执行过程中能够访问模型混淆引擎TA，通过AICPU算子机制及NPU卡内localhost socket进行中转。

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用 “aclnnObfuscationSetupGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnObfuscationSetup”接口执行计算。

- `aclnnStatus aclnnObfuscationSetupGetWorkspaceSize(int32_t fdToClose, int32_t dataType, int32_t hiddenSize, int32_t tpRank, int32_t modelObfSeedId, int32_t dataObfSeedId, int32_t cmd, int32_t threadNum, aclTensor* fd, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnObfuscationSetup(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnObfuscationSetupGetWorkspaceSize

* **参数说明**：

  * fdToClose（int32_t，计算输入）：待关闭的socket连接符，数据类型为INT32，cmd为3时填写本算子在cmd为1时返回的fd，否则填0。
  * dataType（int32_t，计算输入）：代表Tensor数据类型的编号，数据类型为INT32，仅在cmd设置为1或2时需要填写有效值，否则填0。
    * <term>Atlas 推理系列产品</term>：在{0, 1}中选择，0表示FLOAT、1表示FLOAT16
    * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：在{0, 1, 2, 27}中选择，0表示FLOAT、1表示FLOAT16、2表示INT8、27表示BF16
  * hiddenSize（int32_t，计算输入）：隐藏层维度，数据类型为INT32，支持1-10000，仅在cmd设置为1或2时需要填写有效值，否则填0。
  * tpRank（int32_t，计算输入）：TP Rank，数据类型为INT32，支持0-1024，仅在cmd设置为1或2时需要填写有效值，否则填0。
  * modelObfSeedId（int32_t，计算输入）：模型混淆因子id，用于TA从TEE KMC查询模型混淆因子，数据类型为INT32，仅在cmd设置为1或2时需要填写有效值，否则填0。
  * dataObfSeedId（int32_t，计算输入）：数据混淆因子id，用于TA从TEE KMC查询数据混淆因子，数据类型为INT32，仅在cmd设置为1或2时需要填写有效值，否则填0。
  * cmd（int32_t，计算输入）：setup指令编号，在{1, 2, 16}中选择，设置为1时进行普通模式资源初始化、为2时进行高精度模式资源初始化，设置为16时进行资源释放。
  * threadNum（int32_t，计算输入）：CA/TA进行混淆处理使用的线程数。在{1, 2, 3, 4, 5, 6}中选择，仅在cmd设置为1或2时需要填写有效值，否则填0。
  * fd（aclTensor*，计算输出）：socket连接符，数据类型为INT32，1D，shape为1维，[数据格式](../common/数据格式.md)要求为ND。不支持[非连续的Tensor](../common/非连续的Tensor.md)，不支持空Tensor。
  * workspaceSize（uint64_t*，出参）：返回用户需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

* **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_nn.md)。

  ```
  第一段接口会完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的fd是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. fd的数据类型和数据格式不在支持的范围之内。
  ```

## aclnnObfuscationSetup

* **参数说明：**

  * workspace（void*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnObfuscationSetupGetWorkspaceSize获取。
  * executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的Stream。
* **返回值：**
  返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_nn.md)。

## 约束说明
- 确定性计算：
  - aclnnObfuscationSetup默认确定性实现。

- 该接口与[aclnnObfuscationCalculate](./aclnnObfuscationCalculate.md)配套使用，完成PMCC模型混淆功能，使用方式如下：
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
