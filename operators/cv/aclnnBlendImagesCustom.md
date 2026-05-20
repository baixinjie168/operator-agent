# aclnnBlendImagesCustom

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：完成张量rgb、frame和alpha的透明度乘法计算。

- 计算公式：

$$
out_{i*3}=rgb_{i*3} * (alpha_i / 255) + frame_{i*3}*(1 - alpha_i/255)
$$

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnBlendImagesCustomGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnBlendImagesCustom”接口执行计算。

* `aclnnStatus aclnnBlendImagesCustomGetWorkspaceSize(const aclTensor *rgb, const aclTensor *alpha, const aclTensor *frame, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnBlendImagesCustom(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnBlendImagesCustomGetWorkspaceSize

- **参数说明：**
  
  - rgb(aclTensor*, 计算输入): Device侧的aclTensor，数据类型支持UINT8，shape支持HWC(C=3)，与alpha满足[broadcast关系](../common/broadcast关系.md)。只支持连续Tensor,[数据格式](../common/数据格式.md)支持ND。
  - alpha(aclTensor*, 计算输入): Device侧的aclTensor，数据类型支持UINT8，shape支持HWC(C=1)，与rgb满足[broadcast关系](../common/broadcast关系.md)。只支持连续Tensor,[数据格式](../common/数据格式.md)支持ND。
  - frame(aclTensor*, 计算输入): Device侧的aclTensor，数据类型支持UINT8，shape支持HWC(C=3)，与alpha满足[broadcast关系](../common/broadcast关系.md)。只支持连续Tensor,[数据格式](../common/数据格式.md)支持ND。
  - out(aclTensor*, 计算输出): Device侧的aclTensor，数据类型支持UINT8，shape支持HWC(C=3)，与frameshape一致。只支持连续Tensor,[数据格式](../common/数据格式.md)支持ND。
  - workspaceSize(uint64_t\*, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor\*\*, 出参): 返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_cv.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的rgb、alpha、frame或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. rgb、alpha、frame的数据类型和数据格式不在支持的范围之内。
                                       2. rgb、alpha、frame的shape无法做broadcast，rgb和frame支持HWC(C=3), alpha支持HWC(C=1)。
  ```

## aclnnBlendImagesCustom

- **参数说明：**
  - workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnBlendImagesCustomGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参): 指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_cv.md)。

## 约束说明
- 确定性计算：
  - aclnnBlendImagesCustom默认确定性实现

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_cv.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_blend_images_custom.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> rgbShape = {4, 3};
  std::vector<int64_t> alphaShape = {4, 1};
  std::vector<int64_t> frameShape = {4, 3};
  std::vector<int64_t> outShape = {4, 3};

  void* rgbDeviceAddr = nullptr;
  void* alphaDeviceAddr = nullptr;
  void* frameDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* rgb = nullptr;
  aclTensor* alpha = nullptr;
  aclTensor* frame = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> rgbHostData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  std::vector<float> alphaHostData = {255, 255, 255, 255};
  std::vector<float> frameHostData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  std::vector<float> outHostData = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

  ret = CreateAclTensor(rgbHostData, rgbShape, &rgbDeviceAddr, aclDataType::ACL_UINT8, &rgb);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_UINT8, &alpha);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(frameHostData, frameShape, &frameDeviceAddr, aclDataType::ACL_UINT8, &frame);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnBlendImagesCustom第一段接口
  ret = aclnnBlendImagesCustomGetWorkspaceSize(rgb, alpha, frame, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlendImagesCustomGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnBlendImagesCustom第二段接口
  ret = aclnnBlendImagesCustom(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlendImagesCustom failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(rgb);
  aclDestroyTensor(alpha);
  aclDestroyTensor(frame);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(rgbDeviceAddr);
  aclrtFree(alphaDeviceAddr);
  aclrtFree(frameDeviceAddr);
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
