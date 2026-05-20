# aclnnMoeTokenPermuteWithRoutingMapGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- **算子功能**：aclnnMoeTokenPermuteWithRoutingMap的反向传播。
- **计算公式**：

    $$
    permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
    $$

    $$
    capacity = permutedTokenOutputGrad.size(0) / numExperts
    $$

    - probs不为None：
    
    $$
    probsGradOutOptional = zeros(tokens_num, numExperts)
    $$
    
    - paddedMode为true时
    
        $$
        probsGradOutOptional [sortedIndices[i], i/capacity] = permutedProbsOutputGradOptional[i]
        $$
    
    - paddedMode为false时
    
        $$
        probsGradOutOptional = maskedscatter(probsGradOutOptional,routingMap,permutedProbsOutputGradOptional)
        $$
    
    - probs为None：
    
        $$
        tokensGradout= zeros(restoreShapeOptional, dtype=permutedTokens.dtype, device=permutedTokens.device)
        $$
        
        $$
        tokensGradout[permuteTokenId[i]] += permutedTokens[outIndex[i]]
        $$

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenPermuteWithRoutingMapGrad”接口执行计算。

* `aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(const aclTensor *permutedTokenOutputGrad, const aclTensor *permutedProbsOutputGradOptional, const aclTensor *sortedIndices, const aclTensor *routingMapOptional, int64_t numExperts, int64_t tokensNum, bool dropAndPad, aclTensor *tokensGradOut, aclTensor *probsGradOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize

- **参数说明：**
  
  - permutedTokenOutputGrad（aclTensor \*，计算输入）：Device侧的aclTensor，正向输出permutedTokens的梯度，要求为一个维度为2D的Tensor，非droppad模式要求shape为一个2D的（tokens_num \* topK_num，hidden_size），droppad模式要求shape为一个2D的（experts_num \* capacity，hidden_size），其中topK_num表示每个token选中的专家数量，capacity表示每个专家选中的token数量。数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)，不支持空tensor。
  - permutedProbsOutputGradOptional（aclTensor \*，计算输入）：Device侧的aclTensor，可选输入，不传则表示不需要计算probsGradOutOptional，非droppad模式要求shape为一个1D的（tokens_num \* topK_num），droppad模式要求shape为一个1D的（experts_num \* capacity），其中topK_num表示每个token选中的专家数量，capacity表示每个专家选中的token数量。数据类型支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。
  - sortedIndices（aclTensor \*，计算输入）：Device侧的aclTensor，非droppad模式要求shape为一个1D的（tokens_num \* topK_num，），数据类型支持INT32，[数据格式](../common/数据格式.md)要求为ND。索引取值范围[0，tokens_num \* topK_num - 1], droppad模式要求shape为一个1D的（experts_num \* capacity），数据类型支持INT32，[数据格式](../common/数据格式.md)要求为ND。索引取值范围[0，experts_num \* capacity - 1]。支持[非连续的Tensor](../common/非连续的Tensor.md)。
  - routingMap（aclTensor \*，计算输入）：Device侧的aclTensor，代表token到expert的映射关系，要求shape为一个2D的（tokens_num，experts_num），数据类型支持INT8、bool。当数据类型为INT8，取值支持0、1，当数据类型为bool，取值支持true、false，[数据格式](../common/数据格式.md)要求为ND。支持[非连续的Tensor](../common/非连续的Tensor.md)。非droppad模式要求每行中包含topK个true 或 1。
  - experts_num（int64\_t，计算输入）：表示参与运算的专家个数。
  - tokens_num（int64\_t，计算输入）：表示参与运算的token个数。
  - dropAndPad（bool, 计算输入）：true表示开启dropPaddedMode，false表示关闭dropPaddedMode。
  - tokensGradOut（aclTensor\*，计算输出）：输入permutedTokens的梯度，要求是一个2D的Tensor，shape为（tokens_num ，hidden_size）。数据类型同permutedTokenOutputGrad，支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../common/数据格式.md)要求为ND。不支持[非连续的Tensor](../common/非连续的Tensor.md)。
  - probsGradOutOptional（aclTensor\*，计算输出）：输入probs的梯度，可选输出，要求是一个2D的Tensor，shape为（tokens_num，experts_num）。数据类型同permutedProbsOutputGradOptional，支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](../common/数据格式.md)要求为ND。不支持[非连续的Tensor](../common/非连续的Tensor.md)。
  - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。
```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. 输入和输出的Tensor是空指针。
  161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型不在支持的范围内。
                                   2. 输入和输出的Shape不在支持的范围内。
```

## aclnnMoeTokenPermuteWithRoutingMapGrad

- **参数说明：**

    - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize获取。
    - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    - stream（aclrtStream,入参）：指定执行任务的Stream。
- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。

## 约束说明
- 确定性计算：
  - aclnnMoeTokenPermuteWithRoutingMapGrad默认确定性实现。
- 非dropPaddedMode 场景topK_num <= 512
- 不支持混合精度输入，即permutedTokenOutputGrad、permutedProbsOutputGradOptional、tokensGradOut、probsGradOutOptional需要保持相同的数据类型

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_trans.md)。

```Cpp
#include "aclnnop/aclnn_moe_token_permute_with_routing_map_grad.h"
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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


template <typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T>& hostData)
{
    size_t fileSize = 1;
    for (int64_t i : shape){
        fileSize *= i; 
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    // 获取文件大小
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    hostData.reserve(fileSize);
    if (file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << "读取文件失败" << std::endl;
        return 1;
    }
    file.close();
    return true;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        LOG_PRINT("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        LOG_PRINT("Write file Failed.");
        return false;
    }

    return true;
}
void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{

    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造

    int64_t num_token = 4096;
    int64_t hidden_size = 7168;
    int64_t num_expert = 256;
    int64_t num_capacity = 16;
    std::vector<float> permuted_output_grad_Data(num_expert * num_capacity * hidden_size, 0);
    std::vector<int64_t> permuted_output_grad_Shape = {num_expert * num_capacity, hidden_size};
    void* permuted_output_grad_Addr = nullptr;
    aclTensor* permuted_output_grad = nullptr;

    ret = CreateAclTensor(permuted_output_grad_Data, permuted_output_grad_Shape, &permuted_output_grad_Addr,
                          aclDataType::ACL_FLOAT, &permuted_output_grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> permutedProbsOutputGradOptional(num_expert * num_capacity, 0.1);
    std::vector<int64_t> permutedProbsOutputGradOptionalShape = {num_expert * num_capacity};
    void* permutedProbsOutputGrad_Addr = nullptr;
    aclTensor* ppermutedProbsOutputGrad = nullptr;
    ret = CreateAclTensor(permutedProbsOutputGradOptional, permutedProbsOutputGradOptionalShape,
                          &permutedProbsOutputGrad_Addr, aclDataType::ACL_FLOAT, &ppermutedProbsOutputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int> sortedIndicesData(num_expert * num_capacity, 0);
    std::vector<int64_t> sortedIndicesShape = {num_expert * num_capacity};
    void* sortedIndicesAddr = nullptr;
    aclTensor* sortedIndices = nullptr;
    ReadFile("./sortedIndices.bin", sortedIndicesShape, sortedIndicesData);
    ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr, aclDataType::ACL_INT32,
                          &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<char> routingMapOptionalData(num_token * num_expert, 1);
    std::vector<int64_t> routingMapOptionalShape = {num_token , num_expert};
    void* routingMapOptionalAddr = nullptr;
    aclTensor* proutingMapOptional = nullptr;

    ret = CreateAclTensor(routingMapOptionalData, routingMapOptionalShape, &routingMapOptionalAddr,
                          aclDataType::ACL_INT8, &proutingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData(num_token * hidden_size, 0.0f);
    std::vector<int64_t> outShape = {num_token, hidden_size};
    // std::vector<int64_t> outShape = {num_token};
    void* outAddr = nullptr;
    aclTensor* out = nullptr;

    ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData2(num_token * num_expert, 0.0f);
    std::vector<int64_t> outShape2 = {num_token, num_expert};
    void* outAddr2 = nullptr;
    aclTensor* out2 = nullptr;

    ret = CreateAclTensor(outData2, outShape2, &outAddr2, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnMoeTokenPermuteGrad第一段接口
    ret = aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(permuted_output_grad, ppermutedProbsOutputGrad, sortedIndices, proutingMapOptional, num_expert, num_token, true,
                                                                 out, out2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnMoeTokenPermuteWithRoutingMapGrad第二段接口
    ret = aclnnMoeTokenPermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(outShape, &outAddr);
    PrintOutResult(outShape2, &outAddr2);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(permuted_output_grad);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(out);

    // 7. 释放device资源
    aclrtFree(permuted_output_grad_Addr);
    aclrtFree(sortedIndicesAddr);
    aclrtFree(outAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
