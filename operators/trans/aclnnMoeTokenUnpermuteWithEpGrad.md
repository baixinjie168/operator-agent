# aclnnMoeTokenUnpermuteWithEpGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- **接口功能**：aclnnMoeTokenUnpermuteWithEp的反向传播。
- **计算公式**：

  - probs非None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$：
    - 首先计算unpermutedTokens：
      - 当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时：
      
        $$
        unpermutedTokens[i] = permutedTokensOptional[sortedIndices[i]-rangeOptional[0]]
        $$

      - 否则：
        
        $$
        unpermutedTokens[i] = 0
        $$
    
    - 接着计算：

      $$
      unpermutedTokens = unpermutedTokens.reshape(-1, topkNum, hiddenSize)
      $$
      
      $$
      unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
      $$
      
      $$
      probsGrad = \sum_{k=0}^{topkNum}(unpermutedTokens_{i,j,k})
      $$
    
    -  最后，当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时：
      
      $$
      permutedTokensGradOut[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
      $$

  - probs为None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$：
    -  当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时：
    
    $$
    permutedTokensGradOut[sortedIndices[i]-rangeOptional[0]] = unpermutedOutputGrad[i]
    $$

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenUnpermuteWithEpGrad”接口执行计算。

* `aclnnStatus aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices, const aclTensor *permutedTokensOptional, const aclTensor *probsOptional, bool paddedMode, const aclIntArray *restoreShapeOptional, const aclIntArray *rangeOptional, int64_t topkNum, const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenUnpermuteWithEpGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize

- **参数说明：**
  
  -   unpermutedTokensGrad（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的unpermutedTokensGrad，正向输出unpermutedTokens的梯度，要求为一个维度为2D的Tensor，shape为（tokens_num，hidden_size），tokens_num代表token个数，hidden_size代表token的维度大小，数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
  -   sortedIndices（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的sortedIndices，要求shape为一个1D的（tokens_num \* topkNum），数据类型支持INT32，[数据格式](../common/数据格式.md)要求为ND。索引取值范围[0，tokens_num \* topkNum - 1]。支持[非连续的Tensor](../common/非连续的Tensor.md)
  -   permutedTokensOptional（aclTensor \*，计算输入）：Device侧的aclTensor，可选输入，公式中的permutedTokensOptional，要求为一个维度为2D的Tensor，shape为（tokens_num \* topkNum，hidden_size），其中topkNum <= 512，数据类型支持同unpermutedTokensGrad，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)
  -   probsOptional（aclTensor \*，计算输入）：Device侧的aclTensor，可选输入，公式中的probsOptional，要求shape为一个2D的（tokens_num，topkNum），数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../common/数据格式.md)要求为ND。当probs传时，topkNum等于probs第2维；当probs不传时，topkNum=1。支持[非连续的Tensor](../common/非连续的Tensor.md)
  -   paddedMode（bool, 计算输入）：公式中的paddedMode，true表示开启paddedMode，false表示关闭paddedMode，paddedMode解释见restoreShapeOptional参数。目前仅支持false。
  -   restoreShapeOptional（aclIntArray\*，计算输入）：公式中的restoreShapeOptional，当paddedMode为true后生效，否则不会对其进行操作。当paddedMode为true以后，此为unpermutedTokens的shape。当前仅支持nullptr。
  -   rangeOptional（aclIntArray \*，计算输入）：公式中的rangeOptional，ep切分的有效范围，要求rangeOptional[0]代表的起始位置小于rangeOptional[1]代表的结束位置，size为2，为空时不生效。
  -   topkNum（int64\_t，计算输入）：公式中的topkNum，每个token被选中的专家个数。
  -   permutedTokensGradOut（aclTensor \*，计算输出）：输入permutedTokens的梯度，公式中的permutedTokensGradOut，要求是一个2D的Tensor，shape为（tokens_num \* topkNum，hidden_size）。数据类型同permutedTokensOptional，支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../common/数据格式.md)要求为ND。不支持非连续输出。
  -   probsGradOut（aclTensor \*，计算输出）：可选输出，公式中的probsGradOut，输入probs的梯度，要求是一个2D的Tensor，shape为（tokens_num，topkNum）。数据类型同probsOptional，支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../common/数据格式.md)要求为ND。不支持非连续输出。
  -   workspaceSize（uint64\_t \*，出参）：返回需要在Device侧申请的workspace大小。
  -   executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. 必选输入或必选输出的Tensor是空指针。
  161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型不在支持的范围内。
  561002(ACLNN_ERR_INNER_TILING_ERROR): 1. topkNum > 512
                                        2. 输入和输出的shape不符合要求
                                        3. rangeOptional[1] < rangeOptional[0]
  ```

## aclnnMoeTokenUnpermuteWithEpGrad

- **参数说明：**
  
  -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize获取。
  -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  -   stream（aclrtStream,入参）：指定执行任务的Stream。
- **返回值：**
  
    返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。

## 约束说明
- 确定性计算：
  - aclnnMoeTokenUnpermuteWithEpGrad默认确定性实现。
- topkNum <= 512

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_trans.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_ep_grad.h"
#include <iostream>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> permutedTokensShape = {3, 2};
  std::vector<int64_t> unpermutedTokensGradShape = {1, 2};
  std::vector<int64_t> probsShape = {1, 3};
  std::vector<int64_t> sortedIndicesShape = {3};
  std::vector<int64_t> permutedTokensGradShape = {3, 2};
  std::vector<int64_t> probsGradShape = {1, 3};
  void* permutedTokensDeviceAddr = nullptr;
  void* unpermutedTokensGradDeviceAddr = nullptr;
  void* probsDeviceAddr = nullptr;
  void* sortedIndicesDeviceAddr = nullptr;
  void* permutedTokensGradDeviceAddr = nullptr;
  void* probsGradDeviceAddr = nullptr;

  aclTensor* permutedTokens = nullptr;
  aclTensor* unpermutedTokensGrad = nullptr;
  aclTensor* probs = nullptr;
  aclTensor* sortedIndices = nullptr;
  bool paddedMode = false;
  aclTensor *permutedTokensGrad = nullptr;
  aclTensor *probsGrad = nullptr;

  std::vector<float> permutedTokensHostData = {1, 1, 1, 1, 1, 1};
  std::vector<float> unpermutedTokensGradHostData = {1, 1};
  std::vector<float> probsHostData = {1, 1, 1};
  std::vector<int> sortedIndicesHostData = {0, 1, 2};
  std::vector<float> permutedTokensGradHostData = {0, 0, 0, 0, 0, 0};
  std::vector<float> probsGradHostData = {0, 0, 0};

  ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape,
                        &permutedTokensDeviceAddr, aclDataType::ACL_BF16,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unpermutedTokensGradHostData, unpermutedTokensGradShape, &unpermutedTokensGradDeviceAddr,
                      aclDataType::ACL_BF16, &unpermutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr,
                      aclDataType::ACL_BF16, &probs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(permutedTokensGradHostData, permutedTokensGradShape, &permutedTokensGradDeviceAddr, aclDataType::ACL_BF16,
                        &permutedTokensGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsGradHostData, probsGradShape, &probsGradDeviceAddr, aclDataType::ACL_BF16,
                        &probsGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenUnpermuteWithEpGrad第一段接口
  ret = aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(unpermutedTokensGrad, sortedIndices,permutedTokens, probs, paddedMode, nullptr, nullptr, 1, permutedTokensGrad, probsGrad, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnMoeTokenUnpermuteWithEpGrad第二段接口
  ret = aclnnMoeTokenUnpermuteWithEpGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(permutedTokensGradShape, &permutedTokensGradDeviceAddr);
  PrintOutResult(probsGradShape, &probsGradDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(unpermutedTokensGrad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probs);
  aclDestroyTensor(permutedTokensGrad);
  aclDestroyTensor(probsGrad);

  // 7. 释放device资源
  aclrtFree(permutedTokensDeviceAddr);
  aclrtFree(unpermutedTokensGradDeviceAddr);
  aclrtFree(probsDeviceAddr);
  aclrtFree(sortedIndicesDeviceAddr);
  aclrtFree(permutedTokensGradDeviceAddr);
  aclrtFree(probsGradDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
