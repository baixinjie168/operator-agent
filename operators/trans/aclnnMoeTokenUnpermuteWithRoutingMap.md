# aclnnMoeTokenUnpermuteWithRoutingMap


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：对经过aclnnMoeTokenpermuteWithRoutingMap处理的permutedTokens，累加回原unpermutedTokens。根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘，最后进行累加求和，并输出计算结果。
- **计算公式**：
  
  $$
  topK\_num= permutedTokens.size(0) // routingMapOptional.size(0)
  $$

  $$
  numExperts = probs.size(1)
  $$

  $$
  numTokens = probs.size(0)
  $$

  $$
  capacity = sortedIndices.size(0) // numExperts
  $$

  (1)probs不为None，padMode为true时：

  $$
  permuteProbs  [i//capacity,sortedIndices[i]]=probs[i]
  $$

  $$
  permutedTokens = permutedTokens  * permuteProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (2)probs不为None，padMode为false时:

  $$
  permuteProbs = probs.T.maskedSelect(routingMap.T)
  $$

  $$
  permutedTokens = permutedTokens  * permuteProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$


  (3)probs为None,padMode为true时:

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (4)probs为None,padMode为false时:

  $$
  unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$

## 函数原型
  
  每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenUnpermuteWithRoutingMap”接口执行计算。
  
* `aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(const aclTensor *permutedTokens, const aclTensor *sortedIndices, const aclTensor* routingMapOptional, const aclTensor *probsOptional, bool paddedMode, const aclIntArray *restoreShapeOptional, aclTensor *unpermutedTokens, aclTensor *outIndex, aclTensor *permuteTokenId, aclTensor *permuteProbs, uint64_t *workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMap(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`
  
## aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize
  
  - **参数说明：**
    
    - permutedTokens（aclTensor*，计算输入）：Device侧的aclTensor，输入token，要求为一个维度为2D的Tensor，当paddedMode为false时，shape为（tokens_num * topK_num， hidden_size），当paddedMode为true时，shape为（experts_num* capacity， hidden_size），capacity表示每个专家能够处理的token个数，数据类型支持BFLOAT16、FLOAT16、FLOAT，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
    - sortedIndices（aclTensor \*，计算输入）：Device侧的aclTensor，非droppad模式要求shape为一个1D的（tokens_num \* topK_num，），数据类型支持INT32，[数据格式](../common/数据格式.md)要求为ND，索引取值范围[0，tokens_num \* topK_num - 1]。droppad模式要求shape为一个1D的（experts_num \* capacity），数据类型支持INT32，[数据格式](../common/数据格式.md)要求为ND，索引取值范围[0，tokens_num - 1]。支持[非连续的Tensor](../common/非连续的Tensor.md)。
    - routingMapOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选输入，当输入probsOptional为空指针时不需要此输入，应该传入空指针。计算公式中的routingMapOptional，代表对应位置的Token是否被对应专家处理，要求shape为一个2D的（tokens_num，experts_num），数据类型支持INT8、bool。当数据类型为INT8，取值支持0、1，当数据类型为bool，取值支持true、false，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
    - probsOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选输入，当不需要时为空指针。计算公式中的probsOptional，代表对应位置的Token被对应专家处理后的结果在最终结果中的权重，shape与routingMapOptional相同，数据类型与permutedTokens相同，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
    - paddedMode（bool, 计算输入）：host侧的BOOL。可选输入，支持取值为false和true。true表示开启paddedMode，false表示关闭paddedMode，开启paddedMode时，输出outIndex、permuteTokenId的shape为（experts_num* capacity，），关闭paddedMode时，每个token固定被topK_num个专家处理，输出outIndex、permuteTokenId的shape为（tokens_num \* topK_num，）。
    - restoreShapeOptional（aclIntArray*，计算输入）：host侧的aclIntArray。支持的数据类型为INT64, size大小为2。为unpermutedTokens的shape。
    - unpermutedTokens（aclTensor*，计算输出）：Device侧的aclTensor，正向输出结果，计算公式中的unpermutedTokens，要求为一个维度为2D的Tensor，shape为（tokens_num，hidden_size），数据类型支持BFLOAT16、FLOAT16、FLOAT，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
    - outIndex（aclTensor*，计算输出）：Device侧的aclTensor，计算公式中的outIndex，当paddedMode为false时，要求shape为一个1D的（tokens_num * topK_num，），索引取值范围[0，tokens_num * topK_num - 1]。当paddedMode为true时，要求shape为一个1D的（experts_num* capacity，）。索引取值范围[0，experts_num* capacity- 1]。数据类型支持INT32，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
    - permuteTokenId（aclTensor*，计算输出）：Device侧的aclTensor，计算公式中的permuteTokenId，当paddedMode为false时，要求shape为一个1D的（tokens_num * topK_num，）。当paddedMode为true时，要求shape为一个1D的（experts_num* capacity，）。索引取值范围[0，tokens_num - 1]。数据类型支持INT32，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
    - permuteProbs（aclTensor \*，计算输出）：Device侧的aclTensor, 计算公式中的permuteProbs，表示输出经过排序后的probs，shape支持1D维度。数据类型同probsOptional，[数据格式](../common/数据格式.md)要求为ND。
    - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
    - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
  - **返回值：**
    
    aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。
    
    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001(ACLNN_ERR_PARAM_NULLPTR): 1. 必选输入或输出的Tensor是空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 1. 输入或输出的数据类型不在支持的范围内。
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. topK_num > 512。
                                          2. topK_num大于experts_num。
                                          3. capacity大于tokens_num。
                                          4. 输入或输出的shape不符合要求。
    ```
  
## aclnnMoeTokenUnpermuteWithRoutingMap
  
  - **参数说明：**
    
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的Stream。
  - **返回值：**
    
      返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。
  
## 约束说明
- 确定性计算：
  - aclnnMoeTokenUnpermuteWithRoutingMap默认确定性实现。
- topkNum <= 512, pad模式为false时routingMap中每行为1或true的个数固定且小于`512`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_trans.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_routing_map.h"
#include <iostream>
#include <vector>

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
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> permutedTokensShape = {2, 2};
    std::vector<int64_t> sortedIndicesShape = {2};
    std::vector<int64_t> routingMapOptionalShape = {2, 2};
    std::vector<int64_t> probsShape = {2, 2};
    std::vector<int64_t> unpermutedTokensShape = {2, 2};
    std::vector<int64_t> outIndexShape = {2};
    std::vector<int64_t> permuteTokenIdShape = {2};
    std::vector<int64_t> permuteProbsShape = {2};

    void* permutedTokensDeviceAddr = nullptr;
    void* sortedIndicesDeviceAddr = nullptr;
    void* routingMapOptionalDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* unpermutedTokensDeviceAddr = nullptr;
    void* outIndexDeviceAddr = nullptr;
    void* permuteTokenIdDeviceAddr = nullptr;
    void* permuteProbsDeviceAddr = nullptr;
    //in
    aclTensor* permutedTokens = nullptr;
    aclTensor* sortedIndices = nullptr;
    aclTensor* routingMapOptional = nullptr;
    aclTensor* probs = nullptr;
    aclTensor* unpermutedTokens = nullptr;
    aclTensor* outIndex = nullptr;
    aclTensor* permuteTokenId = nullptr;
    aclTensor* permuteProbs = nullptr;
    bool padMode = true;
    std::vector<int64_t> restoreShapeOptionalData = {2, 2};
    aclIntArray *restoreShapeOptional = aclCreateIntArray(restoreShapeOptionalData.data(), restoreShapeOptionalData.size());

    //构造数据
    std::vector<float> permutedTokensHostData = {1.0, 1.0, 1.0, 1.0};
    std::vector<int> sortedIndicesHostData = {1, 1};
    std::vector<char> routingMapOptionalHostData = {1, 1, 1, 1};
    std::vector<float> probsHostData = {1, 1, 1, 1};
    
    std::vector<float> unpermutedTokensHostData = {0, 0, 0, 0};
    std::vector<int> outIndexHostData = {0, 0};
    std::vector<int> permuteTokenIdHostData = {0, 0};
    std::vector<float> permuteProbsHostData = {0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr, aclDataType::ACL_FLOAT, &permutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr, aclDataType::ACL_INT32, &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(routingMapOptionalHostData, routingMapOptionalShape, &routingMapOptionalDeviceAddr, aclDataType::ACL_INT8, &routingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(unpermutedTokensHostData, unpermutedTokensShape, &unpermutedTokensDeviceAddr, aclDataType::ACL_FLOAT, &unpermutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr, aclDataType::ACL_INT32, &permuteTokenId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteProbsHostData, permuteProbsShape, &permuteProbsDeviceAddr, aclDataType::ACL_FLOAT, &permuteProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeTokenUnpermuteWithRoutingMap第一段接口
    ret = aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(permutedTokens, sortedIndices, routingMapOptional, probs, padMode, restoreShapeOptional, 
                                                               unpermutedTokens, outIndex, permuteTokenId, permuteProbs, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMoeTokenUnpermuteWithRoutingMap(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapfailed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto unpermutedTokensSize = GetShapeSize(unpermutedTokensShape);
    std::vector<float> unpermutedTokensData(unpermutedTokensSize, 0);
    ret = aclrtMemcpy(unpermutedTokensData.data(), unpermutedTokensData.size() * sizeof(unpermutedTokensData[0]), unpermutedTokensDeviceAddr, unpermutedTokensSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < unpermutedTokensSize; i++) {
        LOG_PRINT("unpermutedTokensData[%ld] is: %f\n", i, unpermutedTokensData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(permutedTokens);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(routingMapOptional);
    aclDestroyTensor(probs);
    aclDestroyTensor(unpermutedTokens);
    aclDestroyTensor(outIndex);
    aclDestroyTensor(permuteTokenId);
    aclDestroyTensor(permuteProbs);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(permutedTokensDeviceAddr);
    aclrtFree(sortedIndicesDeviceAddr);
    aclrtFree(routingMapOptionalDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(unpermutedTokensDeviceAddr);
    aclrtFree(outIndexDeviceAddr);
    aclrtFree(permuteTokenIdDeviceAddr);
    aclrtFree(permuteProbsDeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
