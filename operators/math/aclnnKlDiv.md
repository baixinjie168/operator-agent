# aclnnKlDiv

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/kl_div_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

* 算子功能：计算KL散度
* 计算公式：
  * 定义loss_pointwise，保存中间结果。
    
    $$
    loss\_pointwise_i=\begin{cases}
    NaN & \text{ if }&logTarget=false \text{ and } target_i <= 0,  \\
    target_i * \left ( \log{(target_i)}- self_i  \right )  & \text{ if }& logTarget=false， \\
    \exp^ {target_i} * \left ( target_i- self_i \right )  & \text{ else. }
    \end{cases}
    $$
    
  * out计算公式为：
  
    $$
    out=\begin{cases}
    \bar{loss\_pointwise}  & \text{ if }& reduction= 1, \\
    \sum loss\_pointwise & \text{ elif }& reduction= 2,\\
    \frac{\sum loss\_pointwise}{self.size(0)} & \text{ elif }& reduction= 3,\\
    loss\_pointwise & \text{ else. }
    \end{cases}
    $$

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnKlDivGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnKlDiv”接口执行计算。

* `aclnnStatus aclnnKlDivGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, bool logTarget, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnKlDiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnKlDivGetWorkspaceSize

* **参数说明**：
  * self (aclTensor*,计算输入)：公式中的`self`。Device侧的aclTensor。且数据类型与target的数据类型需满足数据类型推导规则（参见[互推导关系](../common/互推导关系.md)），shape需要与target满足[broadcast关系](../common/broadcast关系.md)。支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT、FLOAT16
  * target (aclTensor*, 计算输入)：公式中的`target`。且数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](../common/互推导关系.md)），shape需要与self满足[broadcast关系](../common/broadcast关系.md)。支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT、FLOAT16
  * reduction (int64_t，计算输入): 公式中的`reduction`，指定计算完loss_pointwise之后的操作。
    - 0：none，表示不做reduction操作。
    - 1：mean，表示计算loss_pointwise的均值。
    - 2：sum，表示对loss_pointwise求和。
    - 3：batchmean，表示计算批次的平均损失，与Kullback_Leibler散度的数学定义一致。
  * logTarget(bool，计算输入): 指定传入的target数据是否已经做过log操作。
  * out (aclTensor*，计算输出)：公式中的`out`。数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](../common/互推导关系.md)）。当`reduction`为 0 时，shape需要与self和target满足[broadcast关系](../common/broadcast关系.md)。当`reduction`不为0时，shape固定为(1,)。[数据格式](../common/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、
     * <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：FLOAT、FLOAT16
  * workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。
* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

  ````
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的 self 或 target 或 out 是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. self 或 target 或 out 的数据类型不在支持的范围之内。
                                       2. self 的 shape 不能与 target broadcast。 
                                       3. self 或 target 不能转换成 out 的数据类型。
                                       4. self 或 target 的shape dim大于8。
  ````

## aclnnKlDiv

- **参数说明**：
  
  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnKlDivGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnKlDiv默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kl_div.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> targetShape = {4, 2};
  std::vector<int64_t> outShape = {1};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> targetHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData = {1};
  int64_t reduction = 1;
  bool log_target = false;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnKlDiv第一段接口
  ret = aclnnKlDivGetWorkspaceSize(self, target, reduction, log_target, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKlDivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnKlDiv第二段接口
  ret = aclnnKlDiv(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKlDiv failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
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
