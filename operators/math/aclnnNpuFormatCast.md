# aclnnNpuFormatCast

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/npu_format_cast)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 完成ND←→[NZ](../common/数据格式.md)的转换功能。C0是[NZ](../common/数据格式.md)数据格式最后一维的大小。计算方法C0 = 32B / ge::GetSizeByDataType(static_cast<aclDataType>additionalDtype)。
    - 完成NCDHW←→[NDC1HWC0](../common/数据格式.md)、NCDHW←→[FRACTAL_Z_3D](../common/数据格式.md)的转换功能。其中，C0与微架构强相关，该值等于cube单元的size，例如16。C1是将C维度按照C0切分：C1=C/C0， 若结果不整除，最后一份数据需要padding到C0。计算方法C0 = 32B srcDataType（例如FP16为2byte）
- **计算流程**：`aclnnNpuFormatCastCalculateSizeAndFormat`根据输入张量srcTensor、数据类型`additionalDtype`和目标张量的数据格式dstFormat计算出转换后目标张量dstTensor的shape和实际数据格式，用于构造dstTensor，然后调用`aclnnNpuFormatCast`把srcTensor转换为实际数据格式的目标张量dstTensor。
## 函数原型

必须先调用`aclnnNpuFormatCastCalculateSizeAndFormat`计算出dstTensor的shape和实际数据格式，再调用[两段式接口](../common/两段式接口.md)。 两段式接口先调用`aclnnNpuFormatCastGetWorkSpaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnNpuFormatCast`接口执行计算。

- `aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat)`

- `aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnNpuFormatCastCalculateSizeAndFormat

- **参数说明**


  - srcTensor(aclTensor*, 计算输入)：输入张量，Device侧的aclTensor，输入数据支持连续和[非连续的Tensor](../common/非连续的Tensor.md)
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：[数据格式](../common/数据格式.md)支持ND、NZ、NCDHW、NDC1HWC0、FRACTAL_Z_3D。数据类型支持INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32。当数据格式为ND时，支持的shape维度为[2, 6]。
  - dstFormat(int, 计算输入)：输出张量的数据格式。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：[数据格式](../common/数据格式.md)支持：ND、NZ、NCDHW、NDC1HWC0、FRACTAL_Z_3D。

  - additionalDtype(int, 计算输入)：转换为FRACTAL_NZ数据格式时，推断C0大小所使用的基本数据类型。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：该参数仅支持取值srcTensor的数据类型。

  - dstShape(int64_t**, 出参)：用于输出dstTensor的shape数组的指针。该指针指向的内存由本接口申请，调用者释放。

  - dstShapeSize(uint64_t*, 出参)：用于输出dstTensor的shape数组大小的指针。

  - actualFormat(int*, 出参)：用于输出dstTensor实际数据格式的指针。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当前输出[数据格式](../common/数据格式.md)为ACL_FORMAT_ND(2)、ACL_FORMAT_FRACTAL_NZ(29)、ACL_FORMAT_NCDHW(30)、ACL_FORMAT_NDC1HWC0(32)、ACL_FRACTAL_Z_3D(33)。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

  ```
  入参校验，出现以下场景时报错：
  - 161001 (ACLNN_ERR_PARAM_NULLPTR)：
  1、传入的srcTensor是空指针。
  - 161002 (ACLNN_ERR_PARAM_INVALID)：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    1、srcTensor的数据格式非ND、NZ、NCDHW、NDC1HWC0、FRACTAL_Z_3D，数据类型非INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32。
    2、dstFormat的数据格式非ND、NZ、NCDHW、NDC1HWC0、FRACTAL_Z_3D，数据类型非INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32。
    3、additionalDtype的数据类型非srcTensor的数据类型。
    4、srcTensor的view shape维度不在[2, 6]的范围。(ND→NZ时)
  - 361001(ACLNN_ERR_RUNTIME_ERROR):
  1、产品型号不支持。
  2、转换格式不支持。
  ```

## aclnnNpuFormatCastGetWorkspaceSize

- **参数说明**
  - srcTensor(aclTensor*, 计算输入)：输入张量，Device侧的aclTensor，输入的数据只支持连续的Tensor。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：[数据格式](../common/数据格式.md)支持ND、NZ、NCDHW、NDC1HWC0、FRACTAL_Z_3D。数据类型支持：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32。
  - dstTensor(aclTensor*, 计算输入)：转换后的目标张量，Device侧的aclTensor，只支持连续的Tensor。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：[数据格式](../common/数据格式.md)支持ND、NZ、NCDHW、NDC1HWC0、FRACTAL_Z_3D。数据类型支持INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32。
  - workspaceSize(uint64_t*, 出参)：需要在Device侧申请的workspace的大小。
  - executor(aclOpExecutor**, 出参)：包含算子计算流程的op执行器。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  - 161001 (ACLNN_ERR_PARAM_NULLPTR)：
  1、传入的srcTensor、dstTensor是空指针。
  - 161002 (ACLNN_ERR_PARAM_INVALID)：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    1、srcTensor的数据类型非INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式非ND、NZ、NCDHW、NDC1HWC0、FRACTAL_Z_3D。
    2、dstTensor的数据类型非INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式非ND、NZ、NCDHW、NDC1HWC0、FRACTAL_Z_3D。
    3、srcTensor、dstTensor传入非连续的Tensor。
  - 361001(ACLNN_ERR_RUNTIME_ERROR):
  1、产品型号不支持。
  ```

## aclnnNpuFormatCast

- **参数说明**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint_64, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnNpuFormatCastGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：包含算子计算流程的op执行器。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnNpuFormatCast默认确定性实现。
输入和输出支持以下数据类型组合：
当前不支持的特殊场景:
- srcTensor的数据类型和additionalDtype相同，且类型为FLOAT16或BFLOAT16时，若维度表示为[k, n], 则k为1场景暂不支持。
- 不支持调用当前接口转昇腾亲和[数据格式](../common/数据格式.md)FRACTAL_NZ后, 进行任何能修改张量的操作, 如contiguous、pad、slice等;
- 当srcTensor的shape后两维任意一维度shape等于1场景，也不允许转昇腾亲和[数据格式](../common/数据格式.md)FRACTAL_NZ后再进行任何修改张量的操作, 包括transpose。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

  aclnnNpuFormatCastCalculateSizeAndFormat接口参数：
  | srcTensor | dstFormat                 | additionalDtype              | actualFormat                    |
  | --------- | ------------------------- | ---------------------------- | ------------------------------- |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32      | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32                 | ACL_FORMAT_FRACTAL_NZ(29)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_ND(2) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32   | ACL_FORMAT_ND(2) |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NCDHW(30) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32   | ACL_FORMAT_NCDHW(30) |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NDC1HWC0(32) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32   | ACL_FORMAT_NDC1HWC0(32) |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FRACTAL_Z_3D(33) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32   | ACL_FRACTAL_Z_3D(33) |


  aclnnNpuFormatCastGetWorkspaceSize接口：
  | srcTensor | dstTensor数据类型 | dstTensor数据格式               |
  | --------- | ----------------- | ------------------------------- |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32              | ACL_FORMAT_FRACTAL_NZ(29)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_ND(2)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NCDHW(30)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NDC1HWC0(32)       |
  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FRACTAL_Z_3D(33)       |

  C0计算方法：$C0=\frac{32B}{size\ of\ srcTensor的基础类型}$
    | srcTensor的基础类型 | C0 |
    | --------------- | -- |
    | ACL_FLOAT(0)、ACL_INT32(3)、ACL_UINT32(8)     | 8 |
    | ACL_FLOAT16(1)、ACL_BF16(27)  | 16 |
    | ACL_INT8(2)、ACL_UINT8(4)    | 32 |


当前不支持的特殊场景:
- 不支持调用当前接口转昇腾亲和[数据格式](../common/数据格式.md)FRACTAL_NZ后, 进行任何能修改张量的操作, 如contiguous、pad、slice等;
- 不允许转昇腾亲和[数据格式](../common/数据格式.md)FRACTAL_NZ后再进行任何修改张量的操作, 包括transpose。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例.md)。
  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, const int additionalDtype,  int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat);
  extern "C" aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor);
  extern "C" aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

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

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                aclDataType dataType, aclTensor** tensor, aclFormat format) {
      auto size = hostData.size() * sizeof(T);
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

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                  format, *storageShape, *storageShapeSize, *deviceAddr);
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
      int64_t k = 64;
      int64_t n = 128;
      int64_t srcDim0 = k;
      int64_t srcDim1 = n;
      int dstFormat = 29;
      aclDataType srcDtype = aclDataType::ACL_INT32;
      aclDataType additionalDtype = aclDataType::ACL_FLOAT16;

      std::vector<int64_t> srcShape = {srcDim0, srcDim1};
      void* srcDeviceAddr = nullptr;
      void* dstDeviceAddr = nullptr;
      aclTensor* srcTensor = nullptr;
      aclTensor* dstTensor= nullptr;
      std::vector<int32_t> srcHostData(k * n, 1);
      for (size_t i = 0; i < k; i++) {
          for (size_t j = 0; j < n; j++) {
              srcHostData[i * n + j] = (j + 1) % 128;
          }
      }

      std::vector<int32_t> dstTensorHostData(k * n, 1);

      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;

      // 创建src  aclTensor
      ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, srcDtype, &srcTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // 计算目标tensor的shape和format
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      ret = CreateAclTensorWithFormat(dstTensorHostData, srcShape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnNpuFormatCastGetWorkspaceSize第一段接口
      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // 调用aclnnNpuFormatCastGetWorkspaceSize第二段接口
      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
      auto size = 1;
      for (size_t i = 0; i < dstShapeSize; i++) {
          size *= dstShape[i];
      }

      std::vector<int32_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6. 释放dstShape、aclTensor和aclScalar
      delete[] dstShape;
      aclDestroyTensor(srcTensor);
      aclDestroyTensor(dstTensor);

      // 7. 释放device资源
      aclrtFree(srcDeviceAddr);
      aclrtFree(dstDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例.md)。
  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, const int additionalDtype,  int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat);
  extern "C" aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor);
  extern "C" aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

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
      // 此处修改src的format
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                                  shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                aclDataType dataType, aclTensor** tensor, aclFormat format) {
      auto size = hostData.size() * sizeof(T);
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

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                  format, *storageShape, *storageShapeSize, *deviceAddr);
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

      int dstFormat = 32;
      //此处修改目标format
      aclDataType srcDtype = aclDataType::ACL_INT32;
      int additionalDtype = -1;

      // std::vector<int64_t> srcShape = {srcDim0 , srcDim1};
      int64_t N = 1;
      int64_t C = 17;
      int64_t D = 1;
      int64_t H = 2;
      int64_t W = 2;


      std::vector<int64_t> srcShape = {N, C, D, H, W};
      void* srcDeviceAddr = nullptr;
      void* dstDeviceAddr = nullptr;
      aclTensor* srcTensor = nullptr;
      aclTensor* dstTensor= nullptr;
      std::vector<int32_t> srcHostData(N * C * D * H * W, 1);

      int num = 0;
      for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
              for (int d = 0; d < D; ++d) {
                  for (int h = 0; h < H; ++h) {
                      for (int w = 0; w < W; ++w) {
                          // 按 行主序排布，计算线性索引
                          int index = (((n * C + c) * D + d) * H + h) * W + w;
                          srcHostData[index] = num;
                          num++;
                      }
                  }
              }
          }
      }

      std::vector<int32_t> dstTensorHostData(N * C * D * H * W, 1);

      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;

      // 创建src  aclTensor
      ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, srcDtype, &srcTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;
      std::cout << "init actualFormat = " << actualFormat << std::endl;
      // 计算目标tensor的shape和format
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, dstFormat, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      std::cout << "actualFormat = " << actualFormat << std::endl;
      std::cout << "&dstShape = " << &dstShape << std::endl;
      std::cout << "dstShape = [ ";
      for (int64_t i = 0; i < dstShapeSize; ++i) {
          std::cout << dstShape[i] << " ";
      }
      std::cout << "]" << std::endl;

      ret = CreateAclTensorWithFormat(dstTensorHostData, srcShape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnNpuFormatCastGetWorkspaceSize第一段接口
      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // 调用aclnnNpuFormatCastGetWorkspaceSize第二段接口
      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. (固定写法)同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
      auto size = 1;
      for (size_t i = 0; i < dstShapeSize; i++) {
          size *= dstShape[i];
      }

      std::vector<int32_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6. 释放dstShape、aclTensor和aclScalar
      delete[] dstShape;
      aclDestroyTensor(srcTensor);
      aclDestroyTensor(dstTensor);

      // 7. 释放device资源
      aclrtFree(srcDeviceAddr);
      aclrtFree(dstDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
