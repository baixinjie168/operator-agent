# aclnnGroupedMatmul

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

-   接口功能：实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应shape。根据x、weight、y的Tensor数量支持如下4种场景：

    -   x、weight、y都为多tensor，即每组的数据对应的Tensor是独立的。
    -   x为单tensor，weight/y为多tensor，此时需要通过可选参数group_list说明x在行上的分组情况，如group_list[0]=10说明x的前10行参与第一组矩阵乘计算。
    -   x、weight为多tensor，y为单tensor，此时每组矩阵乘的结果放在同一个Tensor中连续存放。
    -   x、y为单tensor，weight为多tensor，属于前两种情况的组合。

    **说明：** 单tensor指一个tensor list中所有分组的tensor在M轴上合并为1个；否则为多tensor。
-   计算公式：
    - **非量化场景：**

    $$
      y_i=x_i\times weight_i + bias_i
    $$

    - **量化场景：**

    $$
      y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
    $$

    - **反量化场景：**

    $$
      y_i=(x_i\times weight_i + bias_i) * scale_i
    $$

    - **伪量化场景：**

    $$
      y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
    $$

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnGroupedMatmulGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGroupedMatmul”接口执行计算。

* `aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(const aclTensorList* x, const aclTensorList* weight, const aclTensorList* biasOptional, const aclTensorList* scaleOptional, const aclTensorList* offsetOptional, const aclTensorList* antiquantScaleOptional, const aclTensorList* antiquantOffsetOptional, const aclIntArray* groupListOptional, int64_t splitItem, const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnGroupedMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnGroupedMatmulGetWorkspaceSize

- **参数说明：**
  -   x（aclTensorList\*，计算输入）：必选参数，Device侧的aclTensorList，公式中的输入x，[数据格式](../common/数据格式.md)支持ND，支持的最大长度为128个。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、INT8
  -   weight（aclTensorList\*，计算输入）：必选参数，Device侧的aclTensorList，公式中的weight，[数据格式](../common/数据格式.md)支持ND，支持的最大长度为128个。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、INT8
  -   biasOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，公式中的bias，[数据格式](../common/数据格式.md)支持ND，长度与weight相同。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32、INT32
  -   scaleOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，代表量化参数中的缩放因子，[数据格式](../common/数据格式.md)支持ND，长度与weight相同。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持UINT64
  -   offsetOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，代表量化参数中的偏移量，[数据格式](../common/数据格式.md)支持ND，长度与weight相同。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持FLOAT32
  -   antiquantScaleOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，代表伪量化参数中的缩放因子，[数据格式](../common/数据格式.md)支持ND，长度与weight相同，数据类型支持FLOAT16、BFLOAT16。
  -   antiquantOffsetOptional（aclTensorList\*，计算输入）可选参数，Device侧的aclTensorList，代表伪量化参数中的偏移量，[数据格式](../common/数据格式.md)支持ND，长度与weight相同，数据类型支持FLOAT16、BFLOAT16。
  -   groupListOptional（aclIntArray\*，计算输入）：可选参数，Host侧的aclIntArray类型，代表输入和输出M方向的matmul索引情况，数据类型支持INT64，[数据格式](../common/数据格式.md)支持ND，长度与weight相同。需注意：当输出中TensorList的长度为1时，groupListOptional中的最后一个值约束了输出数据的有效部分，groupListOptional中未指定的部分将不会参与更新。
  -   splitItem（int64\_t，计算输入）：整数型参数，代表输出是否要做tensor切分，0/1代表输出为多tensor；2/3代表输出为单tensor。
  -   y（aclTensorList\*，计算输出）：Device侧的aclTensorList，公式中的输出y，[数据格式](../common/数据格式.md)支持ND，支持的最大长度为128个。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32。
  -   workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：
  1.如果传入参数是必选输入、输出或者必选属性，且是空指针。
  2.传入参数weight的元素存在空指针。
  3.传入参数x的元素为空指针，且传出参数y的元素不为空指针。
  4.传入参数x的元素不为空指针，且传出参数y的元素为空指针。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：
  1.x、weight、biasOptional、scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、groupListOptional、splitItem、y的数据类型和数据格式不在支持的范围内。
  2.weight的长度大于128。
  3.若bias不为空，bias的长度不等于weight的长度。
  4.splitItem为2、3的场景，y长度不等于1。
  5.splitItem为0、1的场景，y长度不等于weight的长度，groupListOptional长度不等于weight的长度。
  ```

## aclnnGroupedMatmul

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的Stream。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。

## 约束说明
  - 确定性计算：
    - aclnnGroupedMatmul默认确定性实现。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 非量化场景支持的输入类型为：
      - x为FLOAT16、weight为FLOAT16、biasOptional为FLOAT16、scaleOptional为 空、offsetOptional为空、antiquantScaleOptional为空、 antiquantOffsetOptional为空、y为FLOAT16；
      - x为BFLOAT16、weight为BFLOAT16、biasOptional为FLOAT32、scaleOptional 为空、offsetOptional为空、antiquantScaleOptional为空、 antiquantOffsetOptional为空、y为BFLOAT16；
      - x为FLOAT32、weight为FLOAT32、biasOptional为FLOAT32、scaleOptional为 空、offsetOptional为空、antiquantScaleOptional为空、 antiquantOffsetOptional为空、y为FLOAT32；
    - 量化场景支持的输入类型为：

      - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为UINT64、 offsetOptional为空、antiquantScaleOptional为空、 antiquantOffsetOptional为空、y为INT8；
    - 伪量化场景支持的输入类型为：
      - x为FLOAT16、weight为INT8、biasOptional为FLOAT16、scaleOptional为空，  offsetOptional为空，antiquantScaleOptional为FLOAT16、 antiquantOffsetOptional为FLOAT16、y为FLOAT16；
      - x为BFLOAT16、weight为INT8、biasOptional为FLOAT32、scaleOptional为 空，offsetOptional为空，antiquantScaleOptional为BFLOAT16、 antiquantOffsetOptional为BFLOAT16、y为BFLOAT16；
    - 如果传入groupListOptional，groupListOptional必须为非负递增数列，  groupListOptional长度不能为1。
    - 当前支持的场景：
      支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，y，例，单  多单表示支持x为单tensor，weight多tensor，y单tensor的场景。

      | 支持场景 | 场景限制 |
      |:-------:| :-------|
      | 多多多 |1）仅支持splitItem为0/1<br>2）x中tensor要求维度一致，支持2-6维，weight中  tensor需为2维，y中tensor维度和x保持一致<br>3）若x中存在tensor大于2维，  groupListOptional必须传空<br>4）若x中tensor为2维且传入  groupListOptional，groupListOptional的差值需与x中tensor的第一维一一对 应 |
      | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional，且最后 一个值与x中tensor的第一维相等<br>3）x,weight,y中tensor需为2维<br>4） weight中每个tensor的N轴必须相等 |
      | 单多多 |1）仅支持splitItem为0/1<br>2）必须传groupListOptional， groupListOptional的差值需与y中tensor的第一维一一对应<br>3）x,weight,y中  tensor需为2维 |
      | 多多单 |1）仅支持splitItem为2/3<br>2）x,weight,y中tensor需为2维 <br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional， groupListOptional的差值需与x中tensor的第一维一一对应 |
    - x和weight中每一组tensor的最后一维大小都应小于65536。$x_i$的最后一维指当属 性transpose_x为false时$x_i$的K轴或当transpose_x为true时$x_i$的M轴。  $weight_i$的最后一维指当属性transpose_weight为false时$weight_i$的N轴或当  transpose_weight为true时$weight_i$的K轴。
    - x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值  2147483647。

## 调用示例
调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_trans.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul.h"

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
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    std::vector<T> hostData(size, 0);
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


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}


int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<std::vector<int64_t>> xShape = {{1, 16}, {4, 32}};
    std::vector<std::vector<int64_t>> weightShape= {{16, 24}, {32, 16}};
    std::vector<std::vector<int64_t>> biasShape = {{24}, {16}};
    std::vector<std::vector<int64_t>> yShape = {{1, 24}, {4, 16}};
    void* xDeviceAddr[2];
    void* weightDeviceAddr[2];
    void* biasDeviceAddr[2];
    void* yDeviceAddr[2];
    aclTensorList* x = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* bias = nullptr;
    aclIntArray* groupedList = nullptr;
    aclTensorList* scale = nullptr;
    aclTensorList* offset = nullptr;
    aclTensorList* antiquantScale = nullptr;
    aclTensorList* antiquantOffset = nullptr;
    aclTensorList* y = nullptr;
    int64_t splitItem = 0;

    // 创建x aclTensorList
    ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensorList
    ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensorList
    ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensorList
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmul第一段接口
    ret = aclnnGroupedMatmulGetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmul第二段接口
    ret = aclnnGroupedMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    for (int i = 0; i < 2; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(y);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    for (int i = 0; i < 2; i++) {
        aclrtFree(xDeviceAddr[i]);
        aclrtFree(weightDeviceAddr[i]);
        aclrtFree(biasDeviceAddr[i]);
        aclrtFree(yDeviceAddr[i]);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
  ```
