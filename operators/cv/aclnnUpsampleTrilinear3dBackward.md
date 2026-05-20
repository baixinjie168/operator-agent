# aclnnUpsampleTrilinear3dBackward

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/image/upsample_trilinear3d_backward)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：[aclnnUpsampleTrilinear3d.md](aclnnUpsampleTrilinear3d.md)的反向计算。

- 计算公式：
  - 正向核心算法逻辑：
    1. 将目标图像缩放到和原始图像一样大的尺寸。
    2. 计算缩放之后的目标图像的点，以及前后相邻的原始图像的点。
    3. 分别计算相邻点到对应目标点的权重，按照权重相乘累加即可得到目标点值。

  - 具体计算逻辑：
    缩放方式分为角对齐和边对齐，角对齐表示按照原始图片左上角像素中心点对齐，边对齐表示按照原始图片左上角顶点及两条边对齐，在计算缩放系数和坐标位置时存在差异。则有以下公式：

    $$
    scale\_d =\begin{cases}
    (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales\_d & alignCorners=false\&scales\_d>0\\
    inputSize[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_h =\begin{cases}
    (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    inputSize[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    (inputSize[4]-1) / (outputSize[2]-1) & alignCorners=true \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    inputSize[4] / outputSize[2] & alignCorners=false
    \end{cases}
    $$

    因此，对于output的某个方向上的点p(x,y,z)，映射回原始图像中的点记为q(x',y',z')，则有关系：

    $$
    x' =\begin{cases}
    x * scale\_d & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_d-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    y' =\begin{cases}
    y * scale\_h & alignCorners=true \\
    MAX(0,{(y+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    z' =\begin{cases}
    z * scale\_w & alignCorners=true \\
    MAX(0,{(z+0.5)*scale\_w-0.5}) & alignCorners=false
    \end{cases}
    $$

    - 记：

      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

      $$
      y_{0} =int(y'),y_{1} =int(y')+1, lambdb_{0} = y_{1}-y', lambdb_{1} =   1-lambdb_{0}
      $$

      $$
      z_{0} =int(z'),z_{1} =int(z')+1, lambdc_{0} = z_{1}-z', lambdc_{1} =   1-lambdc_{0}
      $$

    - 则有以下公式：

      $$
      {V(p_{x, y, z})} = {V(p_{x0, y0, z0})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x0, y0, z1})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x0, y1, z0})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x0, y1, z1})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{1}} + {V(p_{x1, y0, z0})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x1, y0, z1})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x1, y1, z0})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x1, y1, z1})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{1}} 
      $$

    - 假设：正向插值的输出图像out $(x, y, z)$受原图像input $(x_i, y_j, z_k)$影响，则有:
  
      $$
      gradInput(x_i,y_j,z_k) += gradOut(x,y,z) * lambd(x_i,y_j,z_k)
      $$

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleTrilinear3dBackward”接口执行计算。

```Cpp
aclnnStatus aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize(
  const aclTensor   *gradOut, 
  const aclIntArray *outputSize, 
  const aclIntArray *inputSize, 
  bool              alignCorners, 
  double            scalesD, 
  double            scalesH, 
  double            scalesW, 
  aclTensor         *gradInput, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnUpsampleTrilinear3dBackward(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOut</td>
      <td>输入</td>
      <td>表示反向计算的梯度Tensor，对应公式中的`gradOut`。</td>
      <td><ul><li>不支持空Tensor。</li><li>当数据格式为ND时，默认按照NCDHW格式处理。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>NCDHW、NDHWC、ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize</td>
      <td>输入</td>
      <td>表示输入`gradOut`在D、H和W维度上的空间大小，对应公式中的`outputSize`。</td>
      <td>size为3，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize</td>
      <td>输入</td>
      <td>表示输出`gradInput`分别在N、C、D、H和W维度上的空间大小，对应公式中的`inputSize`。</td>
      <td>size为5，且最后三个元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners</td>
      <td>输入</td>
      <td>表示是否对齐角像素点，对应公式中的`alignCorners`。</td>
      <td>如果为true，则输入和输出张量的角像素点会被对齐，否则不对齐。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesD</td>
      <td>输入</td>
      <td>表示输出`gradInput`的depth维度乘数，对应公式中的`scales_d`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH</td>
      <td>输入</td>
      <td>表示输出`gradInput`的height维度乘数，对应公式中的`scales_h`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW</td>
      <td>输入</td>
      <td>表示输出`gradInput`的width维度乘数，对应公式中的`scales_w`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`gradInput`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape在N、C、D、H和W维度上的大小需与`inputSize`中给定的N、C、D、H和W维度上的空间大小一致。</li><li>数据类型和数据格式与入参`gradOut`的数据类型和数据格式保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>NCDHW、NDHWC、ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas 训练系列产品</term>：
  
    参数`gradOut`和`gradInput`的数据类型不支持BFLOAT16。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_cv.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的gradOut、outputSize、inputSize或gradInput是空指针。</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>gradOut的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradOut和gradInput的数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradOut的维度不为5维。</td>
    </tr>
    <tr>
      <td>outputSize的size不等于3。</td>
    </tr>
    <tr>
      <td>outputSize的某个元素值不大于0。</td>
    </tr>
    <tr>
      <td>inputSize的size不等于5。</td>
    </tr>
    <tr>
      <td>inputSize最后三个元素中某个元素值不大于0。</td>
    </tr>
    <tr>
      <td>gradOut与inputSize在N、C维度上的size不同。</td>
    </tr>
    <tr>
      <td>gradOut在D、H、W维度上的size与outputSize[0]、outputSize[1]、outputSize[2]不一致。</td>
    </tr>
    <tr>
      <td>gradInput的shape与inputSize[0]、inputSize[1]、inputSize[2]、inputSize[3]和inputSize[4]不一致。</td>
    </tr>
  </tbody></table>

## aclnnUpsampleTrilinear3dBackward

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：

  **aclnnStatus**：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_cv.md)。

## 约束说明

- 参数`gradOut`、`gradInput`的shape约束：
  - 每个维度的取值小于等于2^20。
  - 参数`gradInput`的N轴和C轴与`gradOut`保持一致。
  - 内存占用需小于60G。内存占用的计算公式如下：

    $$
    N * C * (gradOut\_D * gradOut\_H * gradOut\_W + gradInput\_D * gradInput\_H * gradInput\_W + gradOut\_D * gradOut\_H * gradInput\_W + gradOut\_D * gradInput\_H * gradInput\_W) * sizeof(float) < 60 * 1024 * 1024 * 1024
    $$

    其中：
    - N代表输入和输出的N轴。
    - C代表输入和输出的C轴。
  - N \* C \* gradOut_D \* gradOut_H < 2^31
  - gradInput_W * gradInput_H < 2^31
- 参数outputSize的D轴、H轴、W轴与参数scalesD、scalesH、scalesW，在使用时二选一，即：
  - 当alignCorners为True时：
    - outputSize对应轴的值等于1，scales对应轴的值为0。
    - 其他情况下使用入参inputSize和入参outputSize中对应轴的参数值，且：$scales=(inputSize-1)/(outputSize-1)$。  
  - 当alignCorners为False时：
    - 当入参scalesD或入参scalesH或入参scalesW的值小于等于0时，使用入参outputSize中对应轴的参数值，即：$scales=(inputSize/outputSize)$。
    - 当入参scalesD或入参scalesH或入参scalesW的值大于0时，使用入参scalesD、入参scalesH、入参scalesW的参数值，即outputSize对应轴的值为$floor(inputSize\_D * scalesD)$，或者$floor(inputSize\_H * scalesH)$，或者$floor(inputSize\_W * scalesW)$。
- 确定性计算：
  - aclnnUpsampleTrilinear3dBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_cv.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_trilinear_3d_backward.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW, shape.data(),
        shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradOutShape = {2, 2, 2, 2, 2};
    std::vector<int64_t> gradInputShape = {2, 2, 1, 1, 1};
    void* gradOutDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> gradInputHostData = {2.0, 2, 2, 2};
    std::vector<int64_t> outputSizeData = {2, 2, 2};
    std::vector<int64_t> inputSizeData = {2, 2, 1, 1, 1};
    bool alignCorners = false;
    double scalesD = 0.0;
    double scalesH = 0.0;
    double scalesW = 0.0;
    // 创建gradOut aclTensor
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradInput aclTensor
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const aclIntArray* outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    const aclIntArray* inputSize = aclCreateIntArray(inputSizeData.data(), inputSizeData.size());
    CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnUpsampleTrilinear3dBackward第一段接口
    ret = aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize(
        gradOut, outputSize, inputSize, alignCorners, scalesD, scalesH, scalesW, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnUpsampleTrilinear3dBackward第二段接口
    ret = aclnnUpsampleTrilinear3dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleTrilinear3dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOut);
    aclDestroyTensor(gradInput);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
