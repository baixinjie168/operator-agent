# aclnnGridSampler2D

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/image/grid_sample)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：提供一个输入Tensor以及一个对应的grid网格，然后根据grid中每个位置提供的坐标信息，将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。
- 计算公式：
  
  输入input、grid网格、输出output的尺寸如下：

  $$
  input: (N, C, H_{in}, W_{in})\\
  grid: (N, H_{out}, W_{out}, 2)\\
  output: (N, C, H_{out}, W_{out})
  $$

  其中input、grid、output中的N是一致的，input和output中的C是一致的，grid和output中的$H_{out}$、$W_{out}$是一致的，grid最后一维大小为2，表示input像素位置信息为(x, y)，会将x和y的取值范围归一化到[-1, 1]之间，(-1, 1)表示左上角坐标，(1, 1)表示右下角坐标。
  - 对于超出范围的坐标，会根据paddingMode进行不同处理：

    - paddingMode=0，表示对越界位置用0填充。
    - paddingMode=1，表示对越界位置用边界值填充。
    - paddingMode=2，表示对越界位置用边界值的对称值填充。

  - 对input采样时，会根据interpolationMode进行不同处理：

    - interpolationMode=0，表示取(x, y)周围四个坐标的加权平均值。
    - interpolationMode=1，表示取input中距离(x, y)最近的坐标值。
    - interpolationMode=2，表示取(x, y)周围十六个坐标的加权平均值。

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnGridSampler2DGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGridSampler2D”接口执行计算。

```Cpp
aclnnStatus aclnnGridSampler2DGetWorkspaceSize(
  const aclTensor *input,
  const aclTensor *grid,
  int64_t          interpolationMode,
  int64_t          paddingMode,
  bool             alignCorners,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGridSampler2D(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGridSampler2DGetWorkspaceSize

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
      <td>input</td>
      <td>输入</td>
      <td>进行插值计算的输入张量，对应公式中描述的`input`。</td>
      <td><ul><li>支持空Tensor。</li><li>支持shape为(N, C, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>in</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>in</em>)。H*W < INT32的最大值。`input`的shape最后两维的维度值不能为0。</li></ul></td>
      <td>FLOAT32、FLOAT16、DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>输入</td>
      <td>采样的网络，对应公式中描述的`grid`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与入参`input`的数据类型一致。</li><li>支持shape为(N, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>, 2)，且N与入参`input`的shape中的N一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、DOUBLE</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>interpolationMode</td>
      <td>输入</td>
      <td>表示插值模式，对应公式描述中的`interpolationMode`。</td>
     <td>支持0：bilinear（双线性插值）、1：nearest（最邻近插值）、2：bicubic（双三次插值）三种模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddingMode</td>
      <td>输入</td>
      <td>表示填充模式，即当(x, y)取值超过输入特征图采样范围时，返回一个特定值。对应公式描述中的`paddingMode`。</td>
      <td>支持0：zeros、1：border、2：reflection三种模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners</td>
      <td>输入</td>
      <td>表示设定特征图坐标与特征值的对应方式，设定为true时，特征值位于像素中心。设定为false时，特征值位于像素的角点。</td>
      <td>设定为true时，特征值位于像素中心。设定为false时，特征值位于像素的角点。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>插值计算的最终输出结果，对应公式中描述的`output`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与input的数据类型一致。</li><li>支持shape为(N, C, <em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>, <em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>)，且N、C与input的shape中的N、C一致，<em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>、<em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>与grid的shape中的<em style='font-size: 14px'>H</em><em style='font-size: 8px'>out</em>、<em style='font-size: 14px'>W</em><em style='font-size: 8px'>out</em>一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、DOUBLE</td>
      <td>ND</td>
      <td>4</td>
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
  
    入参`interpolationMode`不支持插值模式2：bicubic（双三次插值）。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  
    入参`interpolationMode`，仅当input数据类型为FLOAT32或者FLOAT16时支持2：bicubic（双三次插值）。

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
      <td>传入的input、grid或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>input、grid、out的数据类型不在支持的范围之内或数据类型不一致。</td>
    </tr>
    <tr>
      <td>interpolationMode或paddingMode的值不在支持范围内。</tr>
    <tr>
      <td>interpolationMode为bicubic时，input、grid、out的数据类型不是FLOAT32或者FLOAT16。</td>
    </tr>
    <tr>
      <td>input、grid、out的维度关系不匹配。</td>
    </tr>
    <tr>
      <td>input最后两维为空。</td>
    </tr>
  </tbody></table>

## aclnnGridSampler2D

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGridSampler2DGetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_cv.md)。

## 约束说明

- 参数`input`、`grid`、`out`的数据格式只支持(N, C, H, W)，当输入其他数据格式时，默认按照(N, C, H, W)格式处理。
- 输入`input`的（H轴的大小 * W轴的大小） < INT32的最大值。
- grid的输入值*图片（长或宽）大于24位的二进制数（16777216），采样点可能存在误差，精度可能产生偏差。
- 如果grid含有大量超过[-1, 1]范围的数据，使用zeros或者border的填充策略时，计算结果中的值会大量重复。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 如果grid存在超出[-1, 1]范围的数据，使用bicubic插值时，小值域数据计算可能存在误差，精度可能产生偏差。
  - 使用bilinear或者bicubic插值时，针对FLOAT16数据类型，需要使用workspace内存。
- <term>Atlas 训练系列产品</term>：使用bilinear插值时，针对FLOAT16数据类型，需要使用workspace内存。
- 确定性计算：
  - aclnnGridSampler2D默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_cv.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grid_sampler2d.h"

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
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
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
    int64_t interpolationMode = 0;
    int64_t paddingMode = 0;
    bool alignCorners = false;
    std::vector<int64_t> inputShape = {1, 1, 5, 8};
    std::vector<int64_t> gridShape = {1, 3, 3, 2};
    std::vector<int64_t> outShape = {1, 1, 3, 3};
    void* inputDeviceAddr = nullptr;
    void* gridDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* grid = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> inputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    std::vector<float> gridHostData = {-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建grid aclTensor
    ret = CreateAclTensor(gridHostData, gridShape, &gridDeviceAddr, aclDataType::ACL_FLOAT, &grid);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnGridSampler2D第一段接口
    ret = aclnnGridSampler2DGetWorkspaceSize(
        input, grid, interpolationMode, paddingMode, alignCorners, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2DGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGridSampler2D第二段接口
    ret = aclnnGridSampler2D(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler2D failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(grid);
    aclDestroyTensor(out);

    // 7. 释放Device资源，需要根据具体API的接口定义修改
    aclrtFree(inputDeviceAddr);
    aclrtFree(gridDeviceAddr);
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
