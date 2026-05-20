# aclnnBackgroundReplace

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：
将输入的新的背景图片与已有图片进行融合，通过掩码的方式将背景替换为新的背景。

- 计算公式：
  
  $$
  out = bkg * (1 - mask) + src * mask
  $$

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnBackgroundReplaceGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBackgroundReplace”接口执行计算。

* `aclnnStatus aclnnBackgroundReplaceGetWorkspaceSize(const aclTensor* bkg, const aclTensor* src, const aclTensor* mask, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnBackgroundReplace(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnBackgroundReplaceGetWorkspaceSize

- **参数说明：**
  * bkg(aclTensor*, 计算输入):Device侧的aclTensor，数据类型支持UINT8、FLOAT16，shape支持HWC(C=1、3)。只支持连续Tensor，[数据格式](../common/数据格式.md)支持ND。
  * src(aclTensor*, 计算输入): Device侧的aclTensor，数据类型支持UINT8、FLOAT16，shape支持HWC(C=1、3)。只支持连续Tensor，[数据格式](../common/数据格式.md)支持ND。
  * mask(aclTensor*, 计算输入):Device侧的aclTensor，数据类型支持FLOAT16，shape支持HWC(C=1)。只支持连续Tensor，[数据格式](../common/数据格式.md)支持ND。
  * out(aclTensor*, 计算输出): Device侧的aclTensor，数据类型支持UINT8、FLOAT16，shape支持HWC(C=1、3)，数据类型和shape与输入背景图片bkg一致，只支持连续Tensor, [数据格式](../common/数据格式.md)支持ND。
  * workspaceSize(uint64_t \*, 出参): 返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \*\*, 出参): 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_cv.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的bkg、src、mask或out为空指针
161002 (ACLNN_ERR_PARAM_INVALID)：1. bkg、src、mask或out的数据类型不在支持的范围内。
                                  2. bkg、src、mask或out的数据格式不在支持的范围内。
                                  3. bkg、src或out的数据类型不一致。
                                  4. bkg、src、mask或out的数据格式不一致。

```

## aclnnBackgroundReplace

- **参数说明：**
  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnBackgroundReplaceGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_cv.md)。

## 约束说明
- 确定性计算：
  - aclnnBackgroundReplace默认确定性实现

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_cv.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_background_replace.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> bkgShape = {4, 2};
  std::vector<int64_t> srcShape = {4, 2};
  std::vector<int64_t> maskShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* bkgDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  void* maskDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* bkg = nullptr;
  aclTensor* src = nullptr;
  aclTensor* mask = nullptr;
  aclTensor* out = nullptr;
  std::vector<uint8_t> bkgHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint8_t> srcHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> maskHostData = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // 创建bkg aclTensor
  ret = CreateAclTensor(bkgHostData, bkgShape, &bkgDeviceAddr, aclDataType::ACL_UINT8, &bkg);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建src aclTensor
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_UINT8, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mask aclTensor
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_FLOAT16, &mask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBackgroundReplace接口调用示例
  // 3. 调用CANN算子库API, 需要修改为具体的API名称
  // 调用aclnnBackgroundReplace第一段接口
  ret = aclnnBackgroundReplaceGetWorkspaceSize(bkg, src, mask, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBackgroundReplaceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBackgroundReplace第二段接口
  ret = aclnnBackgroundReplace(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBackgroundReplace failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<uint8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(bkg);
  aclDestroyTensor(src);
  aclDestroyTensor(mask);
  aclDestroyTensor(out);

  // 7. 释放Device资源，需要根据具体API的接口定义修改
  aclrtFree(bkgDeviceAddr);
  aclrtFree(srcDeviceAddr);
  aclrtFree(maskDeviceAddr);
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
