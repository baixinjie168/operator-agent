# aclnnRingAttentionUpdateV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：将两次FlashAttention的输出根据其不同的softmax的max和sum更新。**与[RingAttentionUpdate](aclnnRingAttentionUpdate.md)接口的区别是：在输入排布为TND的场景下，原RingAttentionUpdate接口中的softmax相关输入的数据排布为BNS8，RingAttentionUpdateV2 接口支持传入字符串参数 inputSoftmaxLayout，用于控制 softmax 相关输入的数据排布是否与attention保持一致（即采用 TND 排布）。**
- 计算公式：

$$
softmax\_max = max(prev\_softmax\_max, cur\_softmax\_max)
$$

$$
softmax\_sum = prev\_softmax\_sum * exp(prev\_softmax\_max - softmax\_max) + cur\_softmax\_sum * exp(cur\_softmax\_max - softmax\_max)
$$

$$
attn\_out = prev\_attn\_out * exp(prev\_softmax\_max - softmax\_max) * prev\_softmax\_sum / softmax\_sum + cur\_attn\_out * exp(cur\_softmax\_max - softmax\_max) * cur\_softmax\_sum / softmax\_sum
$$

## 函数原型

- 每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用“aclnnRingAttentionUpdateV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRingAttentionUpdateV2”接口执行计算。

```cpp
aclnnStatus aclnnRingAttentionUpdateV2GetWorkspaceSize(
  const aclTensor *prevAttnOut, 
  const aclTensor *prevSoftmaxMax, 
  const aclTensor *prevSoftmaxSum, 
  const aclTensor *curAttnOut, 
  const aclTensor *curSoftmaxMax, 
  const aclTensor *curSoftmaxSum, 
  const aclTensor *actualSeqQlenOptional, 
  char *inputLayoutOptional, 
  char *inputSoftmaxLayoutOptional, 
  const aclTensor *attnOutOut, 
  const aclTensor *softmaxMaxOut, 
  const aclTensor *softmaxSumOut, 
  uint64_t *workspaceSize, 
  aclOpExecutor **executor)
```

```cpp
aclnnRingAttentionUpdateV2(
  void *workspace, 
  uint64_t workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream stream)
```

## aclnnRingAttentionUpdateV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1603px"><colgroup>
  <col style="width: 238px">
  <col style="width: 123px">
  <col style="width: 232px">
  <col style="width: 402px">
  <col style="width: 193px">
  <col style="width: 120px">
  <col style="width: 149px">
  <col style="width: 146px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>prevAttnOut</td>
      <td>输入</td>
      <td>公式中的prev_attn_out。</td>
      <td>第一次FlashAttention的输出，输入shape和inputLayoutOptional属性保持一致。<br>当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>prevSoftmaxMax</td>
      <td>输入</td>
      <td>公式中的prev_softmax_max。</td>
      <td>第一次FlashAttention的softmax的max结果，输入shape为(B,N,S,8)或(T,N,8)，最后一维8个数字相同，且需要为正数。<br>B为batch size，N为head number，S为sequence length，T为time。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>prevSoftmaxSum</td>
      <td>输入</td>
      <td>公式中的prev_softmax_sum。</td>
      <td>第一次FlashAttention的softmax的sum结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>curAttnOut</td>
      <td>输入</td>
      <td>公式中的cur_attn_out。</td>
      <td>第二次FlashAttention的输出，数据类型和输入shape和prevAttnOut保持一致。<br>当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>curSoftmaxMax</td>
      <td>输入</td>
      <td>公式中的cur_softmax_max。</td>
      <td>第二次FlashAttention的softmax的max结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>curSoftmaxSum</td>
      <td>输入</td>
      <td>公式中的cur_softmax_sum</td>
      <td>第二次FlashAttention的softmax的sum结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>输入</td>
      <td>从0开始的sequence length的累加。</td>
      <td>当数据排布inputLayoutOptional为TND时，需要传入该参数，这是一个从0开始递增至T的整数aclTensor。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputLayoutOptional</td>
      <td>输入</td>
      <td>prevAttnOut和curAttnOut的数据排布。</td>
      <td>当前支持“TND”和“SBH”。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSoftmaxLayoutOptional</td>
      <td>输入</td>
      <td>prevSoftmaxMax、prevSoftmaxSum、curSoftmaxMax和curSoftmaxSum数据排布。</td>
      <td>当输入数据排布inputLayoutOptional为TND时生效，当前支持空字符串、“SBH”和“TND”。<br>通过此开关控制是否对softmaxMax相关输入做转置操作。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOutOut</td>
      <td>输出</td>
      <td>公式中的attn_out。</td>
      <td>通过两次结果更新后的输出，数据类型和输出shape和prevAttnOut保持一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>输出</td>
      <td>公式中的softmax_max，通过两次结果更新后的softmax的max。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>与prevSoftmaxMax保持一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>输出</td>
      <td>公式中的softmax_sum，通过两次结果更新后的softmax的sum。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>与prevSoftmaxMax保持一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。

  第一段接口完成入参校验，出现以下场景时报错：
 
  <table style="undefined;table-layout: fixed; width: 1182px"><colgroup>
  <col style="width: 313px">
  <col style="width: 119px">
  <col style="width: 750px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的prevAttnOut、prevSoftmaxMax、prevSoftmaxSum、curAttnOut、curSoftmaxMax、curSoftmaxSum、attnOutOut、softmaxMaxOut、softmaxSumOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>prevAttnOut、prevSoftmaxMax、prevSoftmaxSum、curAttnOut、curSoftmaxMax、curSoftmaxSum、attnOutOut、softmaxMaxOut、softmaxSumOut数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>prevAttnOut、prevSoftmaxMax、prevSoftmaxSum、curAttnOut、curSoftmaxMax、curSoftmaxSum、attnOutOut、softmaxMaxOut、softmaxSumOut的shape不支持。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>当actualSeqQlenOptional有输入时，输入数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>当inputSoftmaxLayoutOptional输入值不在支持范围之内。</td>
    </tr>
  </tbody>
  </table>

## aclnnRingAttentionUpdateV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1049px"><colgroup>
  <col style="width: 167px">
  <col style="width: 118px">
  <col style="width: 764px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceAddGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_trans.md)。

## 约束说明

- 确定性计算：
  - aclnnRingAttentionUpdateV2默认确定性实现。
- 当inputLayoutOptional为“TND”时，prevAttnOut的最后一个维度需要为64的倍数。
- 当inputLayoutOptional为“TND”时，actualSeqQlenOptional为必填。
- 当inputLayoutOptional为“TND”时，请注意N和D的大小，限制为：N<=256, D<=768。
- **当inputLayoutOptional为“TND”时，inputSoftmaxLayoutOptional才生效。inputSoftmaxLayoutOptional只支持三种输入：空字符串、“SBH”、“TND”**

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../common/编译与运行样例_trans.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ring_attention_update.h"

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
  // 1. (固定写法)device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t batchNum = 1;
  int64_t headNum = 1;
  int64_t seqSize = 2;
  int64_t headDim = 64;
  int64_t headSize = headNum * headDim;

  std::vector<int64_t> prevAttnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> prevSoftmaxMaxShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> prevSoftmaxSumShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> curAttnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> curSoftmaxMaxShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> curSoftmaxSumShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> actualSeqQlenOptionalShape = {batchNum, headNum};

  std::vector<int64_t> attnOutShape = {seqSize, batchNum, headSize};
  std::vector<int64_t> softmaxMaxShape = {batchNum * seqSize, headNum, 8};
  std::vector<int64_t> softmaxSumShape = {batchNum * seqSize, headNum, 8};

  void* prevAttnOutDeviceAddr = nullptr;
  void* prevSoftmaxMaxDeviceAddr = nullptr;
  void* prevSoftmaxSumDeviceAddr = nullptr;
  void* curAttnOutDeviceAddr = nullptr;
  void* curSoftmaxMaxDeviceAddr = nullptr;
  void* curSoftmaxSumDeviceAddr = nullptr;
  void* actualSeqQlenOptionalDeviceAddr = nullptr;

  void* attnOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* prevAttnOut = nullptr;
  aclTensor* prevSoftmaxMax = nullptr;
  aclTensor* prevSoftmaxSum = nullptr;
  aclTensor* curAttnOut = nullptr;
  aclTensor* curSoftmaxMax = nullptr;
  aclTensor* curSoftmaxSum = nullptr;
  aclTensor* actualSeqQlenOptional = nullptr;

  aclTensor* attnOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;

  std::vector<float> prevAttnOutHostData(seqSize * batchNum * headSize, 1);
  std::vector<float> prevSoftmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> prevSoftmaxSumHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> curAttnOutHostData(seqSize * batchNum * headSize, 1);
  std::vector<float> curSoftmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> curSoftmaxSumHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> actualSeqQlenOptionalHostData(batchNum * headNum, 1);

  std::vector<float> attnOutHostData(seqSize * batchNum * headSize, 1);
  std::vector<float> softmaxMaxHostData(batchNum * headNum * seqSize * 8, 1);
  std::vector<float> softmaxSumHostData(batchNum * headNum * seqSize * 8, 1);

  char* inputLayoutOptional = "TND";
  char* inputSoftmaxLayoutOptional = "TND";
  // 创建prevAttnOut aclTensor
  ret = CreateAclTensor(prevAttnOutHostData, prevAttnOutShape, &prevAttnOutDeviceAddr, aclDataType::ACL_FLOAT, &prevAttnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建prevSoftmaxMax aclTensor
  ret = CreateAclTensor(prevSoftmaxMaxHostData, prevSoftmaxMaxShape, &prevSoftmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &prevSoftmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建prevSoftmaxSum aclTensor
  ret = CreateAclTensor(prevSoftmaxSumHostData, prevSoftmaxSumShape, &prevSoftmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &prevSoftmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建curAttnOut aclTensor
  ret = CreateAclTensor(curAttnOutHostData, curAttnOutShape, &curAttnOutDeviceAddr, aclDataType::ACL_FLOAT, &curAttnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建curSoftmaxMax aclTensor
  ret = CreateAclTensor(curSoftmaxMaxHostData, curSoftmaxMaxShape, &curSoftmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &curSoftmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建curSoftmaxSum aclTensor
  ret = CreateAclTensor(curSoftmaxSumHostData, curSoftmaxSumShape, &curSoftmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &curSoftmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建actualSeqQlenOptional aclTensor
  ret = CreateAclTensor(actualSeqQlenOptionalHostData, actualSeqQlenOptionalShape, &actualSeqQlenOptionalDeviceAddr, aclDataType::ACL_INT64, &actualSeqQlenOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建attnOut aclTensor
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建softmaxMax aclTensor
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建softmaxSum aclTensor
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRingAttentionUpdateV2第一段接口
  ret = aclnnRingAttentionUpdateV2GetWorkspaceSize(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, 
                                                 curAttnOut, curSoftmaxMax, curSoftmaxSum, 
                                                 actualSeqQlenOptional, inputLayoutOptional, inputSoftmaxLayoutOptional,
                                                 attnOut, softmaxMax, softmaxSum, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnRingAttentionUpdateV2第二段接口
  ret = aclnnRingAttentionUpdateV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRingAttentionUpdateV2 failed. ERROR: %d\n", ret); return ret);
  // 4. (固定写法)同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto attnOutSize = GetShapeSize(attnOutShape);
  std::vector<float> attnOutResultData(attnOutSize, 0);
  ret = aclrtMemcpy(attnOutResultData.data(), attnOutResultData.size() * sizeof(attnOutResultData[0]), attnOutDeviceAddr, attnOutSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < attnOutSize; i++) {
    LOG_PRINT("attnOutResultData[%ld] is: %f\n", i, attnOutResultData[i]);
  }

  auto softmaxMaxSize = GetShapeSize(softmaxMaxShape);
  std::vector<float> softmaxMaxResultData(softmaxMaxSize, 0);
  ret = aclrtMemcpy(softmaxMaxResultData.data(), softmaxMaxResultData.size() * sizeof(softmaxMaxResultData[0]), softmaxMaxDeviceAddr, softmaxMaxSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < softmaxMaxSize; i++) {
    LOG_PRINT("softmaxMaxResultData[%ld] is: %f\n", i, softmaxMaxResultData[i]);
  }

  auto softmaxSumSize = GetShapeSize(softmaxSumShape);
  std::vector<float> softmaxSumResultData(softmaxSumSize, 0);
  ret = aclrtMemcpy(softmaxSumResultData.data(), softmaxSumResultData.size() * sizeof(softmaxSumResultData[0]), softmaxSumDeviceAddr, softmaxSumSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < softmaxSumSize; i++) {
    LOG_PRINT("softmaxSumResultData[%ld] is: %f\n", i, softmaxSumResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(prevAttnOut);
  aclDestroyTensor(prevSoftmaxMax);
  aclDestroyTensor(prevSoftmaxSum);
  aclDestroyTensor(curAttnOut);
  aclDestroyTensor(curSoftmaxMax);
  aclDestroyTensor(curSoftmaxSum);
  aclDestroyTensor(actualSeqQlenOptional);
  aclDestroyTensor(attnOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(prevAttnOutDeviceAddr);
  aclrtFree(prevSoftmaxMaxDeviceAddr);
  aclrtFree(prevSoftmaxSumDeviceAddr);
  aclrtFree(curAttnOutDeviceAddr);
  aclrtFree(curSoftmaxMaxDeviceAddr);
  aclrtFree(curSoftmaxSumDeviceAddr);
  aclrtFree(actualSeqQlenOptionalDeviceAddr);
  aclrtFree(attnOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
