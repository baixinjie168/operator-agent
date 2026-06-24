# HTML报告模板参考（明亮风格）

生成HTML报告时，请严格遵循以下结构和样式规范。

## 报告整体结构

```
header (明亮渐变色标题栏)
  |
nav-bar (固定导航栏, sticky top, 白色背景)
  |
summary-bar (摘要统计, 浅色卡片)
  |
main-content (白色/浅灰背景)
  |-- operator-info (算子信息卡片)
  |-- section#sec-param-complete (维度1：参数完整性)
  |-- section#sec-param-attr (维度2：参数属性正确性)
  |-- section#sec-constraint-complete (维度3：约束完整性)
  |-- section#sec-constraint-correct (维度4：约束正确性)
  |-- conclusion-section#sec-conclusion (综合结论)
  |
script (导航栏高亮)
```

## 导航栏定义

导航栏共5个项，每个项包含名称和数量徽标：

| 序号 | 名称 | href | 徽标class | 说明 |
|------|------|------|-----------|------|
| 1 | 参数完整性 | #sec-param-complete | badge-total | 参数是否有遗漏或多余 |
| 2 | 参数属性 | #sec-param-attr | badge-total | type/format/dtype/dimensions等字段正确性 |
| 3 | 约束完整性 | #sec-constraint-complete | badge-warn | Markdown中有但JSON缺失的约束 |
| 4 | 约束正确性 | #sec-constraint-correct | badge-total | 每条约束的四项子检查结果 |
| 5 | 结论 | #sec-conclusion | 无 | 综合分析结论 |

## 明亮风格颜色编码

| 状态 | 主色 | 背景色 | 文字色 | 用途 |
|------|------|--------|--------|------|
| 通过 | #22c55e | #dcfce7 | #166534 | 检查通过的项目 |
| 警告 | #f59e0b | #fef3c7 | #92400e | 需要关注但不一定是错误 |
| 失败 | #ef4444 | #fee2e2 | #991b1b | 明确的错误或矛盾 |
| 信息 | #3b82f6 | #dbeafe | #1e40af | 可优化建议 |
| 页头 | #3b82f6 to #8b5cf6 | 渐变 | #ffffff | 顶部标题栏 |

## CSS核心样式（明亮风格）

完整的内联CSS需要包含以下关键规则：

- body: font-family系统字体栈, background #f8fafc, color #1e293b
- .header: linear-gradient(135deg, #3b82f6, #8b5cf6), padding 28px 32px, color #fff
- .nav-bar: position sticky top 0, z-index 100, background #ffffff, border-bottom 1px solid #e2e8f0, flex, overflow-x auto
- .nav-item: padding 14px 20px, font-size 14px, color #64748b, border-bottom 3px solid transparent
- .nav-item.active: color #3b82f6, border-bottom-color #3b82f6, font-weight 600
- .nav-badge: border-radius 10px, font-size 11px, font-weight 600
- .badge-pass: background #dcfce7, color #166534
- .badge-warn: background #fef3c7, color #92400e
- .badge-fail: background #fee2e2, color #991b1b
- .badge-info: background #dbeafe, color #1e40af
- .badge-total: background #f1f5f9, color #475569
- .summary-bar: background #ffffff, padding 16px 32px, flex, gap 32px
- .summary-card .num: font-size 24px, font-weight 700
- .summary-card .label: color #94a3b8, font-size 13px
- .main-content: max-width 1280px, margin 0 auto, padding 28px 32px
- .section: scroll-margin-top 60px, background #ffffff, border-radius 12px, border 1px solid #e2e8f0
- .section-header: padding 18px 24px, background #fafbfc, border-bottom 1px solid #e2e8f0
- .section-header h3: font-size 16px, color #1e293b
- .constraint-table: width 100%, border-collapse collapse, font-size 13px
- .constraint-table th: background #f8fafc, padding 12px 16px, font-size 12px, color #475569, border-bottom 2px solid #e2e8f0
- .constraint-table td: padding 14px 16px, border-bottom 1px solid #f1f5f9
- .constraint-table tr:hover: background #f8fafc
- .verdict-pass: background #dcfce7, color #166534, border-radius 20px, padding 4px 14px
- .verdict-warning: background #fef3c7, color #92400e, border-radius 20px, padding 4px 14px
- .verdict-fail: background #fee2e2, color #991b1b, border-radius 20px, padding 4px 14px
- .expr-cell code: background #f1f5f9, padding 10px 12px, border-radius 6px, font-family monospace, font-size 12px, border 1px solid #e2e8f0
- .expr-type-badge: background #f1f5f9, color #475569, border-radius 6px, padding 3px 10px, font-family monospace, font-size 11px
- .analysis-cell .check-ok: color #22c55e
- .analysis-cell .check-warn: color #f59e0b
- .analysis-cell .check-fail: color #ef4444
- .analysis-cell .note: background #fffbeb, border 1px solid #fde68a, border-radius 6px, padding 8px 12px, color #92400e
- .product-header: padding 14px 24px, background #eff6ff, border-bottom 1px solid #bfdbfe
- .product-header h3: font-size 14px, color #1d4ed8

## 各章节表格列定义

### 维度1：参数完整性 (sec-param-complete)
检查JSON inputs/outputs与Markdown参数表格的参数是否一致。
表格列: # | 参数名 | Markdown存在 | JSON存在 | 输入/输出 | is_operator_param | 结论 | 分析详情

- 按产品分组，使用 product-header
- 框架参数（workspaceSize, executor）标记为"框架参数"不纳入检查
- 衍生参数（is_operator_param=true）允许仅存在于JSON中

### 维度2：参数属性正确性 (sec-param-attr)
按产品平台分组，每组使用 product-header + constraint-table。
表格列: # | 参数名 | 检查字段 | JSON值 | Markdown值 | 结论 | 分析详情

检查字段包括: type, format, is_optional, is_support_discontinuous, dimensions, array_length, dtype, allowed_range_value

每个参数每个字段一行。如果某字段对特定参数不适用（如format对非Tensor参数），标记为 N/A 并跳过。

### 维度3：约束完整性 (sec-constraint-complete)
检查Markdown中描述但JSON中缺失的约束。
表格列: # | 来源参数 | Markdown原文 | 预期约束类型 | 结论 | 建议补充的表达式

- 来源包括：参数使用说明、产品特定说明、计算公式、调用示例
- 预期约束类型对应 expr_type 枚举

### 维度4：约束正确性 (sec-constraint-correct)
对JSON中每条约束进行四项子检查。
表格列: # | expr_type | 表达式 | src_text原文 | relation_params | Python语法 | 描述一致性 | 结论 | 分析详情

- expr_type使用 expr-type-badge 显示
- 表达式使用 expr-cell code 显示
- relation_params 列显示完整性检查结果
- Python语法列显示语法有效性检查结果
- 描述一致性列显示表达式与src_text的比对结果
- 分析详情展开每项子检查的具体发现

## 导航栏高亮脚本

页面底部包含IntersectionObserver脚本，实现滚动时自动高亮当前章节。
使用 rootMargin "-60px 0px -80% 0px" 适配sticky导航栏高度。
监听 .section 和 .conclusion-section 元素进入视口，自动切换nav-item的active状态。
