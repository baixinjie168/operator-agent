# aclnnMoeTokenPermuteWithRoutingMap

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明
- **算子功能**：MoE的permute计算，将token和expert的标签作为routingMap传入，根据routingMaps将tokens和可选probsOptional广播后排序
- **计算公式**：
  tokens\_num 为routingMap的第0维大小，expert\_num为routingMap的第1维大小
  dropAndPad为`false`时
  
  $$
  expertIndex=arrange(tokens\_num).expand(expert\_num,-1)
  $$
  
  $$
  sortedIndicesFirst=expertIndex.maskedselect(routingMap.T)
  $$
  
  $$
  sortedIndicesOut=argSort(sortedIndicesFirst)
  $$
    
  $$
  topK = numOutTokens // tokens\_num
  $$
  
  $$
  outToken = topK * tokens\_num
  $$

  $$
  permuteTokens[sortedIndicesOut[i]]=tokens[i//topK]
  $$
  
  $$
  permuteProbsOutOptional=probsOptional.T.maskedselect(routingMap.T)
  $$
  
  dropAndPad为`true`时

  $$
  capacity = numOutTokens // expert\_num
  $$
  
  $$
  outToken = capacity * expert\_num
  $$

  $$
  sortedIndicesOut = argsort(routingMap.T,dim=-1)[:, :capacity]
  $$
  
  $$
  permutedTokensOut = tokens.index_select(0, sorted_indices)
  $$
  
  如果probs不是none
  
  $$
  robs\_T\_1D = probsOptional.T.view(-1)
  $$
  
  $$
  indices\_dim0 = arange(num\_experts)
  $$
  
  $$
  indices\_dim1 = sorted_indices.view(expert\_num, capacity)
  $$
  
  $$
  indices\_1D = (indices_dim0 * tokens\_num + indices\_dim1).view(-1)
  $$
  
  $$
  permuteProbsOutOptional = probs\_T\_1D.index_select(0, indices_1D)
  $$


## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用 “aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenPermuteWithRoutingMap”接口执行计算。

* `aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(const aclTensor *tokens, const aclTensor *routingMap, const aclTensor *probsOptional,  int64_t numOutTokens,  bool dropAndPad, aclTensor *permuteTokensOut, aclTensor *permuteProbsOutOptional, aclTensor *sortedIndicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermuteWithRoutingMap(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
## aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize

- **参数说明：**
  
  - tokens（aclTensor \*，计算输入）：Device侧的aclTensor，输入token，公式中的tokens，要求为一个维度为2D的Tensor，shape为 \(tokens\_num, hidden\_size)，数据类型支持BFLOAT16，FLOAT16，FLOAT，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
  - routingMap（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的routingMap，代表token到expert的映射关系，要求shape为一个2D的（tokens_num，experts_num），数据类型支持INT8、BOOL。当数据类型为INT8，取值支持0、1，当数据类型为bool，取值支持true、false，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。非droppad模式要求每行中包含topK个true 或 1。
  - probsOptional（aclTensor \*，计算输入）：Device侧的aclTensor，可选输入probsOptional，公式中的probsOptional，要求元素个数与routingMap相同，当probsOptional为空时，可选输出permuteProbsOutOptional为空，数据类型同tokens。[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
  - numOutTokens（int64\_t，计算输入）：公式中的numOutTokens，用于计算公式中topK 和capacity 的有效输出token数。
  - dropAndPad（bool，计算输入）：公式中的dropAndPad，表示是否开启dropAndPad模式。
  - permutedTokensOut（aclTensor \*，计算输出）：Device侧的aclTensor，公式中的permutedTokensOut，根据indices进行扩展并排序筛选过的tokens，要求是一个2D的Tensor，shape为\(outToken, hidden\_size)，即公式中的outToken。数据类型同tokens，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
  - sortedIndicesOut（aclTensor \*，计算输出）：Device侧的aclTensor，公式中的sortedIndicesOut，permute_tokens和tokens的映射关系， 要求是一个1D的Tensor，Shape为\(outToken\)，即公式中的outToken，数据类型支持INT32，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
  - permuteProbsOutOptional（aclTensor \*，计算输出）：Device侧的aclTensor，公式中的permuteProbsOutOptional，根据indices进行排序并筛选过的probsOptional，Shape为\(outToken\)，即公式中的outToken，数据类型同probsOptional，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
  - workspaceSize（uint64\_t \*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。

  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. 输入和输出的Tensor是空指针。
  161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型不在支持的范围内。
                                   2. 输入输出的shape不符合要求
                                   3. numOutTokens < 0 或 numOutTokens > tokens_num * experts_num
  561002(ACLNN_ERR_INNER_TILING_ERROR): 1. topkNum > 512
  ```

## aclnnMoeTokenPermuteWithRoutingMap

- **参数说明：**
  
  -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize获取。
  -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  -   stream（aclrtStream,入参）：指定执行任务的Stream。
- **返回值：**
  
    返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeTokenPermuteWithRoutingMap默认确定性实现。

- tokens_num和experts_num要求小于`16777215`，pad模式为false时routingMap 中 每行为1或true的个数固定且小于`512`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_trans.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_routing_map.h"
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
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {6, 4};
    std::vector<int64_t> idxOutShape = {6};
    void* xDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    int64_t numTokenOut = 6;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> indicesHostData = {1, 1, 1, 1, 1, 1};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_BOOL, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_BF16, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeTokenPermuteWithRoutingMap第一段接口
    ret = aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(x, indices, nullptr, numTokenOut, padMode, expandedXOut, nullptr, sortedIndicesOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMoeTokenPermuteWithRoutingMap(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapfailed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto sortedIndicesSize = GetShapeSize(idxOutShape);
    std::vector<int> sortedIndicesData(sortedIndicesSize, 0);
    ret = aclrtMemcpy(sortedIndicesData.data(), sortedIndicesData.size() * sizeof(sortedIndicesData[0]), sortedIndicesOutDeviceAddr, sortedIndicesSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sortedIndicesSize; i++) {
        LOG_PRINT("sortedIndicesData[%ld] is: %d\n", i, sortedIndicesData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
