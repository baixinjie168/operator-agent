# aclnnAddLayerNorm

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/add_layer_norm)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：实现AddLayerNorm功能。
- 计算公式：

  $$
  x = x1 + x2 + biasOptional
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+eps}}
  $$

  $$
  y = (x-E(x)) * rstd * gamma + beta
  $$

  其中，E(x)表示均值，Var(x)表示方差，均需要在算子内部计算得到。

## 函数原型

每个算子分为[两段式接口](../common/两段式接口.md)，必须先调用`aclnnAddLayerNormGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddLayerNorm`接口执行计算。

```Cpp
aclnnStatus aclnnAddLayerNormGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *biasOptional,
  double           epsilon,
  bool             additionalOutput,
  const aclTensor *yOut,
  const aclTensor *meanOut,
  const aclTensor *rstdOut,
  const aclTensor *xOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormGetWorkspaceSize

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
      <td>x1</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，将会在算子内做x1 + x2 + biasOptional的计算并对计算结果做层归一化。对应公式中的`x1`。</td>
      <td><ul><li>不支持空Tensor。</li><li>不支持输入的某一维的值为0。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，将会在算子内做x1 + x2 + biasOptional的计算并对计算结果做层归一化。对应公式中的`x2`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape和`x1`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示层归一化中的beta参数。对应公式中的`beta`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape的维度值与`x1`需要norm的维度值相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示层归一化中的gamma参数。对应公式中的`gamma`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape的维度值与`x1`需要norm的维度值相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>可选输入参数，表示AddLayerNorm中加法计算的输入，将会在算子内做x1 + x2 + biasOptional的计算并对计算结果做层归一化。对应公式中的`biasOptional`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape可以和`gamma`/`beta`或`x1`/`x2`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>表示添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。</td>
      <td>取值仅支持1e-5。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additionalOutput</td>
      <td>输入</td>
      <td>表示是否开启x=x1+x2+biasOptional的输出。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>输出</td>
      <td>表示输出LayerNorm计算过程中（x1 + x2 + biasOptional）的结果的均值。对应公式中的`E(x)`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需要与`x1`满足<a href="../common/broadcast关系.md">broadcast关系</a>（前几维的维度和`x1`前几维的维度相同，后面的维度为1，总维度与`x1`维度相同，前几维指`x1`的维度减去gamma的维度，表示不需要norm的维度）。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>输出</td>
      <td>表示输出LayerNorm计算过程中`rstd`的结果。对应公式中的`rstd`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需要与`x1`满足<a href="../common/broadcast关系.md">broadcast关系</a>（前几维的维度和`x1`前几维的维度相同，后面的维度为1，总维度与`x1`维度相同，前几维指`x1`的维度减去gamma的维度，表示不需要norm的维度）。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>表示LayerNorm的结果输出。对应公式中的`y`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需要与输入`x1`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut</td>
      <td>输出</td>
      <td>表示Add的结果输出`x`。对应公式中的`x`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需要与输入`x1`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
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

  - <term>Atlas 推理系列产品</term>：
    - 参数`x1`、`x2`、`beta`、`gamma`、`biasOptional`、`yOut`、`xOut`的数据类型不支持BFLOAT16。
    - 参数`meanOut`、`rstdOut`在当前产品使用场景下无效。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_nn.md)。
  
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="9">561002</td>
      <td>tiling阶段（x1、x2、gamma、beta、yOut、meanOut、rstdOut、xOut）的shape获取失败。</td>
    </tr>
    <tr>
      <td>（x1、gamma）的shape维数大于8或小于0。</td>
    </tr>
    <tr>
      <td>（x1、x2、yOut、meanOut、rstdOut、xOut）的维数不一致。</td>
    </tr>
    <tr>
      <td>x1的维数小于gamma。</td>
    </tr>
    <tr>
      <td>（x1、gamma、meanOut）的任意一个维度等于0。</td>
    </tr>
    <tr>
      <td>（x1、x2、yOut、xOut）的shape不是完全相同的shape。</td>
    </tr>
    <tr>
      <td>（gamma、beta）的shape不是完全相同的shape。</td>
    </tr>
    <tr>
      <td>（meanOut、rstdOut）的shape不是完全相同的shape。</td>
    </tr>
    <tr>
      <td>gamma的维度和x的需要作norm的维度不相同，或meanOut的维度和x的不需要norm的维度不相同，或meanOut的需要norm的维度不为1。</td>
    </tr>
  </tbody></table>


## aclnnAddLayerNorm

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddLayerNormGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../common/aclnn返回码_nn.md)。

## 约束说明

- **功能维度**
  - 数据类型支持
    - <term>Atlas 推理系列产品</term>：x1、x2、beta、gamma、biasOptional支持FLOAT32、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x1、x2、beta、gamma、biasOptional支持FLOAT32、FLOAT16、BFLOAT16。
    - rstdOut、meanOut支持：FLOAT32。
  - 数据格式支持：ND。
  - <term>Atlas 推理系列产品</term>：x1、x2、beta、gamma、biasOptional五个输入的尾轴长度必须大于等于32Bytes。
- **未支持类型说明**
  - DOUBLE：不支持DOUBLE。
- **边界值场景说明**
  - 当输入是Inf时，输出为Inf。
  - 当输入是NaN时，输出为NaN。
- **各产品支持数据类型说明**
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    | x1数据类型 | x2数据类型 | gamma数据类型 | beta数据类型 | biasOptional数据类型 | yOut数据类型 | meanOut数据类型 | rstdOut数据类型 | xOut数据类型 |
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | FLOAT32  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT32  | BFLOAT16 | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | BFLOAT16 | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 |
    | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
  - <term>Atlas 推理系列产品</term>：
    | x1数据类型 | x2数据类型 | gamma数据类型 | beta数据类型 | biasOptional数据类型 | yOut数据类型 | meanOut数据类型 | rstdOut数据类型 | xOut数据类型 |
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT16 |
- 确定性计算：
  - aclnnAddLayerNorm默认确定性实现。
