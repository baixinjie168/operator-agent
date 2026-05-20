# aclnnOneHot

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/one_hot)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    √    |
| <term>Atlas 训练系列产品</term>                              |    √    |


## 功能说明

- 算子功能：对长度为n的输入self， 经过one_hot的计算后得到一个元素数量为n*k的输出out，其中k的值为numClasses。
  输出的元素满足下列公式：
  
  $$
  out[i][j]=\left\{
  \begin{aligned}
  onValue,\quad self[i] = j \\
  offValue, \quad self[i] \neq j
  \end{aligned}
  \right.
  $$

- 示例：

  ```
  示例1：
  self = tensor([0, 1, 2, 0, 1])
  numClasses = 5
  onValue = tensor([1])
  offValue = tensor([0])
  axis=-1
  out的shape为(5,5)
  out = tensor([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0]])

  示例2：
  self = tensor([0, 1, 2, 0, 1])
  numClasses = 1
  onValue = tensor([1])
  offValue = tensor([0])
  axis=-1
  out的shape为(5,1)
  out = tensor([[1],
                [0],
                [0],
                [1],
                [0]])

  示例3：
  self = tensor([0, 1, 2, 0, 1])
  numClasses = 0
  onValue = tensor([1])
  offValue = tensor([0])
  axis=-1
  out的shape为(5,0)
  out = tensor([])

  示例4：
  self = tensor([[1,2,3]]) # shape (1,3)
  numClasses = 4
  onValue = tensor([1])
  offValue = tensor([0])
  axis=1
  out的shape为(1, 4, 3)
  out = tensor([[[0. 0. 0.]
                 [1. 0. 0.]
                 [0. 1. 0.]
                 [0. 0. 1.]]]) # shape (1, 4, 3)
  ```

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnOneHotGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnOneHot”接口执行计算。

- `aclnnStatus aclnnOneHotGetWorkspaceSize(const aclTensor* self, int numClasses, const aclTensor* onValue, const aclTensor* offValue, int64_t axis, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnOneHot(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnOneHotGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*，计算输入)：表示索引张量，公式中的self，Device侧的aclTensor，shape支持1-8维度。支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT32、INT64。
  - numClasses(int，计算输入)：表示类别数，数据类型必须输入INT64。当self为空Tensor时，numClasses的值需大于0；当self不为空Tensor时。numClasses需大于等于0。若numClasses的值为0，则返回空Tensor。如果self存在元素大于numClasses，这些元素会被编码成全offValue。
  - onValue(aclTensor*，计算输入)：表示索引位置的填充值，公式中的onValue，Device侧的aclTensor，shape支持1-8维度，且计算时只使用其中第一个元素值进行计算。数据类型与out一致，支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT32、INT64。
  - offValue(aclTensor*，计算输入)：表示非索引位置的填充值，公式中的offValue，Device侧的aclTensor，shape支持1-8维度，且计算时只使用其中第一个元素值进行计算。数据类型与out一致，支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT32、INT64。
  - axis(int64_t，计算输入)：表示编码向量的插入维度，最小值为-1，最大值为self的维度数。若值为-1，编码向量会往self的最后一维插入。
  - out(aclTensor*，计算输出)：表示one-hot张量，公式中的输出out，Device侧的aclTensor，shape支持1-8维度，且与在self的shape在axis轴插入numClasses后的shape一致，支持[非连续的Tensor](../common/非连续的Tensor.md)，[数据格式](../common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、INT32、INT64。
  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的self、onValue、offValue或out为空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. self、onValue、offValue或out不在支持的数据类型范围之内。
                                        2. onValue、offValue的数据类型与out的数据类型不一致。
                                        3. self为空Tensor，且numClasses小于等于0。
                                        4. self不为空Tensor，且numClasses小于0。
                                        5. axis的值小于-1。
                                        6. axis的值大于self的维度数量。
                                        7. out的维度不比self的维度多1维。
                                        8. out的shape与在self的shape在axis轴插入numClasses后的shape不一致。
                                        9. self、onValue、offValue或out的维度超过8维
  ```

## aclnnOneHot

- **参数说明：**

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnOneHotGetWorkspaceSize获取。
  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnOneHot默认确定性实现。
- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 当offValue的数据类型为INT64时，其首元素取值仅支持0或1。
  - 输入self大小为selfSize，输出out大小为outSize，ub大小为ubSize，当axis取值为0，满足下列条件的场景暂不支持：
    - selfSize * 3 < ubSize - 16K < outSize * 3 / 2


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_one_hot.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {4, 2};
    int numClasses = 4;
    std::vector<int64_t> outShape = {4, 2, 4};
    std::vector<int64_t> onValueShape = {1};
    std::vector<int64_t> offValueShape = {1};
    void *selfDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *onValueDeviceAddr = nullptr;
    void *offValueDeviceAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *out = nullptr;
    aclTensor *onValue = nullptr;
    aclTensor *offValue = nullptr;
    std::vector<int32_t> selfHostData = {0, 1, 2, 3, 3, 2, 1, 0};
    std::vector<int32_t> outHostData = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int32_t> onValueHostData = {1};
    std::vector<int32_t> offValueHostData = {0};
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建onValue aclTensor
    ret = CreateAclTensor(onValueHostData, onValueShape, &onValueDeviceAddr, aclDataType::ACL_INT32, &onValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offValue aclTensor
    ret = CreateAclTensor(offValueHostData, offValueShape, &offValueDeviceAddr, aclDataType::ACL_INT32, &offValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    int64_t axis = -1;
    aclOpExecutor *executor;
    // 调用aclnnoneHot第一段接口
    ret = aclnnOneHotGetWorkspaceSize(self, numClasses, onValue, offValue, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnOnehot第二段接口
    ret = aclnnOneHot(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHot failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(onValue);
    aclDestroyTensor(offValue);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(onValueDeviceAddr);
    aclrtFree(offValueDeviceAddr);
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
