# aclnnMatmulReduceScatterV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

-   **接口功能**:
    aclnnMatmulReduceScatterV2接口是对aclnnMatmulReduceScatter接口的功能扩展，在支持x1和x2输入类型为FLOAT16/BFLOAT16的基础上,
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        -   新增了对低精度数据类型INT8的支持。支持pertoken/perchannel[量化方式](../common/量化介绍.md)。
        -   新增了对FLOAT8_E4M3FN/FLOAT8_E5M2 mixfp8量化的支持[量化方式](../common/量化介绍.md)

-   **计算公式**：
    -   情形1：如果x1和x2数据类型为FLOAT16/BFLOAT16时，入参x1、x2进行matmul计算后，进行ReduceScatter通信。

    $$
    output=ReduceScatter(x1@x2)
    $$

    -   情形2：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的pertensor场景，或者x1和x2数据类型为INT8的perchannel、pertoken场景，且不输出amaxOut，入参x1、x2进行matmul计算和dequant计算后，进行ReduceScatter通信。

    $$
    output=ReduceScatter((x1Scale*x2Scale)*(x1@x2 + bias_{optional}))
    $$

    -   情形3：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的perblock场景，且不输出amaxOut，当x1为(a0, a1)x2为(b0, b1)时x1Scale为(ceildiv(a0, 128), ceildiv(a1, 128))x2Scale为(ceildiv(b0, 128), ceildiv(b1, 128))时，入参x1、x2进行matmul计算和dequant计算后，再进行ReduceScatter通信。
    
    $$
    output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
    $$

    -   情形4：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2的pertensor mixfp8量化场景，且不输出amaxOut，当x1为(a0, a1, 2)x2为(b0, b1, 2)时x1Scale为(a0, ceildiv(a1, 64), 2), x2Scale为(b1, ceildiv(b0, 64), 2)，x1不转置，x2转置时，入参x1、x2进行matmul计算和dequant计算后，再进行ReduceScatter通信。
    
    $$
    output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
    $$
    
## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnMatmulReduceScatterV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMatmulReduceScatterV2”接口执行计算。

* `aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* quantScale, int64_t blockSize, const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, int64_t groupSize, const char* commMode, aclTensor* output, aclTensor* amaxOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnMatmulReduceScatterV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMatmulReduceScatterV2GetWorkspaceSize

-   **参数说明：**
    -   x1（aclTensor\*，计算输入）：Device侧的两维aclTensor，mm左矩阵。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：commMode为aicpu时，数据类型支持FLOAT16、BFLOAT16；commMode为aiv时，数据类型支持FLOAT16、BFLOAT16、INT8, shape为[m, k]，[数据格式](../common/数据格式.md)支持ND。**当前版本仅支持两维输入，且仅支持不转置场景**。
    -   x2（aclTensor\*，计算输入）：Device侧的两维aclTensor，mm左矩阵。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：commMode为aicpu时，数据类型支持FLOAT16、BFLOAT16；commMode为aiv时，数据类型支持FLOAT16、BFLOAT16、INT8。shape为[k, n]，[数据格式](../common/数据格式.md)支持ND。支持通过转置构造的[非连续的Tensor](../common/非连续的Tensor.md)。**当前版本仅支持两维输入**。
    -   bias（aclTensor\*，计算输入）：Device侧的一维aclTensor。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：在commMode为aicpu时，数据类型支持FLOAT16、BFLOAT16，[数据格式](../common/数据格式.md)支持ND。且当前版本仅支持为0的输入。在commMode为aiv时，当前版本仅支持输入nullptr。
    -   x1Scale（aclTensor\*，计算输入）：Device侧的aclTensor，mm左矩阵反量化参数。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：在commMode为aicpu时，仅支持输入nullptr。在commMode为aiv时，数据类型支持FLOAT,[数据格式](../common/数据格式.md)支持ND。当x1和x2数据类型为FLOAT16/BFLOAT16时，仅支持输入为nullptr。在pertoken场景，shape为(m, 1)。
    -   x2Scale（aclTensor\*，计算输入）：Device侧的aclTensor，mm右矩阵反量化参数。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：在commMode为aicpu时，仅支持输入nullptr。在commMode为aiv时，数据类型支持FLOAT、INT64，数据格式支持ND。INT64数据类型仅在output数据类型为FLOAT16场景支持。当x1和x2数据类型为FLOAT16/BFLOAT16时，仅支持输入为nullptr。在perchannel场景，shape为(1, n)。
    -   quantScale（aclTensor\*，计算输入）：Device侧的一维aclTensor，mm输出矩阵量化参数。数据类型支持FLOAT。**当前版本仅支持nullptr**。
    -   blockSize （int64\_t，计算输入）：Host侧的整型，用于表示mm输出矩阵在M轴方向上和N轴方向上可以用于对应方向上的多少个数的量化。blockSize由blockSizeM， blockSizeN，blockSizeK三个值拼接而成，每个值占16位，计算公式为：blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32，mm输出矩阵不涉及K轴，blockSizeK固定为0。**当前版本只支持blockSizeM = blockSizeN = 0**。
    -   group（char\*，计算输入）：Host侧的char，标识列组的字符串。数据类型支持String。通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取, 其中commName即为group。
    -   reduceOp（char\*，计算输入）：Host侧的char，reduce操作类型。数据类型支持String。**当前版本仅支持“sum”**。
    -   commTurn（int64\_t，计算输入）：Host侧的整型，通信数据切分数，即总数据量/单次通信量。数据类型支持INT64。**当前版本仅支持输入0**。
    -   streamMode（int64\_t，计算输入）：Host侧的整型，流模式的枚举，当前只支持枚举值1，数据类型支持INT64。
    -   groupSize（int64_t，计算输入）：用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当前版本仅支持输入为0。
    -   commMode (char\*，计算输入)：Host侧的整型，通信模式。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当前支持两种模式：aicpu和aiv。aicpu模式下使用aicpu完成通信功能，功能等同于aclnnMatmulReduceScatter算子；aiv模式下使用AI VECTOR核完成通信任务。
    -   output（aclTensor\*，计算输出）：Device侧的aclTensor，MatMul计算+ReduceScatter通信的结果。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16，数据格式支持ND。如果x1数据类型为FLOAT16、BFLOAT16时，output数据类型与x1一致。
    -   amaxOutOptional（aclTensor\*，计算输出）：Device侧的一维aclTensor，MatMul计算的最大值结果。**当前版本仅支持nullptr或空tensor**。
    -   workspaceSize（uint64\_t\*，出参）：Device侧的整型，返回需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：Device侧的aclOpExecutor，返回op执行器，包含了算子计算流程。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。
    ```
    161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的x1/x2/output为空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 1. 传入的x1/x2/output/bias(非空场景)/x1Scale(非空场景)/x2Scale(非空场景)/quantScale(非空场景) 的数据格式或数据类型不在支持范围。
    ```

## aclnnMatmulReduceScatterV2

-   **参数说明：**

    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMatmulReduceScatterV2GetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的Stream。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。

## 约束说明

- 确定性计算：
  - aclnnMatmulReduceScatterV2默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 只支持x2矩阵转置/不转置，x1矩阵仅支持不转置场景。
    - 输入x1为2维，其shape为\(m, k\)，m须为卡数rank\_size的整数倍。
    - 输入x2必须是2维，其shape为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
    - bias为1维，shape为\(n,\)。
    - 输出为2维，其shape为\(m/rank\_size, n\), rank\_size为卡数。
    - 不支持空tensor。
    - x1和x2的数据类型需要保持一致。
    - 支持2、4、8卡。
    
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_trans.md)。

```Cpp
#include <iostream>
#include <vector>
#include <thread>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_matmul_reduce_scatter_v2.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

constexpr int DEV_NUM = 2;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i +1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
  };

int LaunchOneThreadMmReduceScatterV2(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
    std::vector<int64_t> x1Shape = {1024, 256};
    std::vector<int64_t> x2Shape = {256, 512};
    std::vector<int64_t> biasShape = {512};
    std::vector<int64_t> x1ScaleShape = {1};
    std::vector<int64_t> x2ScaleShape = {1};
    std::vector<int64_t> outShape = {1024 / DEV_NUM, 512};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *x1ScaleDeviceAddr = nullptr;
    void *x2ScaleDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *x1Scale = nullptr;
    aclTensor *x2Scale = nullptr;
    aclTensor *quantScale = nullptr;
    aclTensor *out = nullptr;
    aclTensor *amaxOut = nullptr;

    int32_t commTurn = 0;
    int32_t streamMode = 1;
    int32_t blockSize = 0;
    int32_t groupSize = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
    long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
    long long outShapeSize = GetShapeSize(outShape);

    std::vector<int8_t> x1HostData(x1ShapeSize, 0);
    std::vector<int8_t> x2HostData(x2ShapeSize, 0);
    std::vector<int32_t> biasHostData(biasShapeSize, 0);
    std::vector<int32_t> x1ScaleHostData(x1ScaleShapeSize, 0);
    std::vector<int32_t> x2ScaleHostData(x2ScaleShapeSize, 0);
    std::vector<int16_t> outHostData(outShapeSize, 0);
    // 创建tensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用第一阶段接口
    ret = aclnnMatmulReduceScatterV2GetWorkspaceSize(
        x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, hcomName, "sum", commTurn, streamMode, groupSize, "aiv",
        out, amaxOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnMatmulReduceScatterV2(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2 failed. ret = %d \n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);
    LOG_PRINT("[INFO] device_%d aclnnMatmulReduceScatterV2 execute successfully.\n", args.rankId);
    // 释放device资源，需要根据具体API的接口定义修改
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (x1Scale != nullptr) {
        aclDestroyTensor(x1Scale);
    }
    if (x2Scale != nullptr) {
        aclDestroyTensor(x2Scale);
    }
    if (quantScale != nullptr) {
        aclDestroyTensor(quantScale);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (amaxOut != nullptr) {
        aclDestroyTensor(amaxOut);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (x1ScaleDeviceAddr != nullptr) {
        aclrtFree(x1ScaleDeviceAddr);
    }
    if (x2ScaleDeviceAddr != nullptr) {
        aclrtFree(x2ScaleDeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = HcclCommDestroy(args.hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }
    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }
    // 初始化集合通信域
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[DEV_NUM];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadMmReduceScatterV2, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
```
