"""
覆盖率报告格式化输出模块

功能：将 CoverageResult 格式化为可读的文本报告。
      支持两种输出格式：
        1. console 格式：终端直接打印，含进度条形图
        2. markdown 格式：适合写入文件或 CI 展示

报告中包含：
  - 基本统计：算子名称、用例数
  - 1-pair 覆盖：每个参数每个属性的逐项覆盖率
  - 2-pair 覆盖：跨参数同属性配对的逐项覆盖率
  - 平均覆盖率汇总
"""

from typing import Dict, List

from .coverage_calculator import CoverageResult
from .attribute_defs import OperatorAttributeDomain


# ============================================================
# 辅助函数
# ============================================================

def _rate_bar(rate: float, width: int = 20) -> str:
    """
    生成进度条形图。

    用 20 个 Unicode 方块字符直观展示覆盖率：
      - █ (filled block): 已覆盖部分
      - ░ (shade block):  未覆盖部分

    例如 75% 覆盖率：
      ███████████████░░░░░
    """
    filled = int(rate * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def _pct(rate: float) -> str:
    """将小数格式的覆盖率转为百分比字符串，保留两位小数。"""
    return f"{rate * 100:6.2f}%"


# ============================================================
# Console 格式报告
# ============================================================

def format_report(
    result: CoverageResult,
    domain: OperatorAttributeDomain,
    case_count: int,
    title: str = "Coverage Report",
) -> str:
    """
    生成 console 格式的覆盖率报告。

    布局：
      1. 标题区（算子名称、用例数）
      2. 1-pair 覆盖表（参数/属性/已覆盖/可能/覆盖率/进度条）
      3. 平均 1-pair 覆盖率
      4. 2-pair 覆盖表（属性对/已覆盖/可能/覆盖率/进度条）
      5. 平均 2-pair 覆盖率
    """
    lines = []
    sep = "=" * 72
    lines.append(sep)
    lines.append(f"  {title}")
    lines.append(f"  算子: {domain.operator_name}  |  用例数: {case_count}")
    lines.append(sep)
    lines.append("")

    # ----------------------------------------------------------
    # 1-pair 覆盖部分
    # ----------------------------------------------------------
    lines.append("─" * 72)
    lines.append("  一、1-pair 覆盖（每参数每属性取值覆盖）")
    lines.append("─" * 72)
    lines.append("")

    header = f"{'参数':<24} {'类型':<10} {'属性':<22} {'已覆盖/可能':<14} {'覆盖率':<10} 分布"
    lines.append(header)
    lines.append("─" * len(header))

    all_one_rates = []
    for param_name in sorted(result.one_pair.keys()):
        param_result = result.one_pair[param_name]
        first_attr = True
        param_type = domain.params[param_name].constraint_type if param_name in domain.params else ""

        for attr_name in sorted(param_result.keys()):
            info = param_result[attr_name]
            label = param_name if first_attr else ""
            type_label = param_type if first_attr else ""
            display = (
                f"{label:<24} {type_label:<10} {attr_name:<22} "
                f"{info['count']:>3}/{info['total']:<8} "
                f"{_pct(info['rate']):<10} {_rate_bar(info['rate'])}"
            )
            lines.append(display)
            first_attr = False
            all_one_rates.append(info["rate"])

        # 参数无任何可枚举属性时的占位
        if first_attr:
            lines.append(f"{param_name:<24} {'(无可枚举属性)':<22}")

    # 平均 1-pair 覆盖率
    if all_one_rates:
        avg = sum(all_one_rates) / len(all_one_rates)
        lines.append("")
        lines.append(f"{'平均 1-pair 覆盖率':>50}  {_pct(avg)}  {_rate_bar(avg)}")

    lines.append("")

    # ----------------------------------------------------------
    # 2-pair 覆盖部分
    # ----------------------------------------------------------
    lines.append("─" * 72)
    lines.append("  二、2-pair 覆盖（跨参数同属性类型配对覆盖）")
    lines.append("─" * 72)
    lines.append("")

    header2 = f"{'属性对':<50} {'已覆盖/可能':<14} {'覆盖率':<10} 分布"
    lines.append(header2)
    lines.append("─" * len(header2))

    all_two_rates = []
    for key in sorted(result.two_pair.keys()):
        info = result.two_pair[key]
        display = (
            f"{key:<50} "
            f"{info['count']:>3}/{info['total']:<8} "
            f"{_pct(info['rate']):<10} {_rate_bar(info['rate'])}"
        )
        lines.append(display)
        all_two_rates.append(info["rate"])

    # 平均 2-pair 覆盖率
    if all_two_rates:
        avg = sum(all_two_rates) / len(all_two_rates)
        lines.append("")
        lines.append(f"{'平均 2-pair 覆盖率':>67}  {_pct(avg)}  {_rate_bar(avg)}")

    lines.append("")

    # ----------------------------------------------------------
    # 跨属性 2-pair 覆盖部分
    # ----------------------------------------------------------
    if result.two_pair_cross:
        lines.append("─" * 72)
        lines.append("  三、跨属性 2-pair 覆盖（不同属性跨参数配对）")
        lines.append("─" * 72)
        lines.append("")

        header3 = f"{'属性对':<50} {'已覆盖/可能':<14} {'覆盖率':<10} 分布"
        lines.append(header3)
        lines.append("─" * len(header3))

        all_cross_rates = []
        for key in sorted(result.two_pair_cross.keys()):
            info = result.two_pair_cross[key]
            display = (
                f"{key:<50} "
                f"{info['count']:>3}/{info['total']:<8} "
                f"{_pct(info['rate']):<10} {_rate_bar(info['rate'])}"
            )
            lines.append(display)
            all_cross_rates.append(info["rate"])

        if all_cross_rates:
            avg = sum(all_cross_rates) / len(all_cross_rates)
            lines.append("")
            lines.append(f"{'平均跨属性 2-pair 覆盖率':>67}  {_pct(avg)}  {_rate_bar(avg)}")
        lines.append("")

    lines.append(sep)

    return "\n".join(lines)


# ============================================================
# Markdown 格式报告
# ============================================================

def format_markdown_report(
    result: CoverageResult,
    domain: OperatorAttributeDomain,
    case_count: int,
    title: str = "Coverage Report",
) -> str:
    """
    生成 Markdown 格式的覆盖率报告。

    适合：
      - 保存为 .md 文件在 Git 仓库中追踪覆盖率变化
      - 在 CI/CD 的 Pipeline 中作为 Artifact 展示
      - 在 Merge Request 中作为评论输出
    """
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **算子**: {domain.operator_name}")
    lines.append(f"- **用例数**: {case_count}")
    lines.append("")

    # ----------------------------------------------------------
    # 1-pair 覆盖表格
    # ----------------------------------------------------------
    lines.append("## 一、1-pair 覆盖（每参数每属性取值覆盖）")
    lines.append("")

    rows = []
    all_one_rates = []
    for param_name in sorted(result.one_pair.keys()):
        param_result = result.one_pair[param_name]
        param_type = domain.params[param_name].constraint_type if param_name in domain.params else ""
        for attr_name in sorted(param_result.keys()):
            info = param_result[attr_name]
            rows.append((param_name, param_type, attr_name, info["count"], info["total"], info["rate"]))
            all_one_rates.append(info["rate"])

    if rows:
        lines.append("| 参数 | 类型 | 属性 | 已覆盖 | 可能取值 | 覆盖率 |")
        lines.append("|------|------|------|-------:|--------:|------:|")
        for p, pt, a, c, t, r in rows:
            lines.append(f"| {p} | {pt} | {a} | {c} | {t} | {_pct(r)} |")
        lines.append("")
        avg = sum(all_one_rates) / len(all_one_rates)
        lines.append(f"**平均 1-pair 覆盖率**: {_pct(avg)}")
    else:
        lines.append("_无可用数据_")

    lines.append("")

    # ----------------------------------------------------------
    # 2-pair 覆盖表格
    # ----------------------------------------------------------
    lines.append("## 二、2-pair 覆盖（跨参数同属性类型配对覆盖）")
    lines.append("")

    rows2 = []
    all_two_rates = []
    for key in sorted(result.two_pair.keys()):
        info = result.two_pair[key]
        rows2.append((key, info["count"], info["total"], info["rate"]))
        all_two_rates.append(info["rate"])

    if rows2:
        lines.append("| 属性对 | 已覆盖 | 可能组合 | 覆盖率 |")
        lines.append("|--------|-------:|--------:|------:|")
        for k, c, t, r in rows2:
            lines.append(f"| {k} | {c} | {t} | {_pct(r)} |")
        lines.append("")
        avg = sum(all_two_rates) / len(all_two_rates)
        lines.append(f"**平均 2-pair 覆盖率**: {_pct(avg)}")
    else:
        lines.append("_无可用数据_")
    lines.append("")

    # ---- 跨属性 2-pair ----
    if result.two_pair_cross:
        lines.append("## 三、跨属性 2-pair 覆盖（不同属性跨参数配对）")
        lines.append("")
        rows3 = []
        all_rates3 = []
        for key in sorted(result.two_pair_cross.keys()):
            info = result.two_pair_cross[key]
            rows3.append((key, info["count"], info["total"], info["rate"]))
            all_rates3.append(info["rate"])
        if rows3:
            lines.append("| 属性对 | 已覆盖 | 可能组合 | 覆盖率 |")
            lines.append("|--------|-------:|--------:|------:|")
            for k, c, t, r in rows3:
                lines.append(f"| {k} | {c} | {t} | {_pct(r)} |")
            lines.append("")
            avg = sum(all_rates3) / len(all_rates3) if all_rates3 else 0.0
            lines.append(f"**平均跨属性 2-pair 覆盖率**: {_pct(avg)}")
            lines.append("")

    return "\n".join(lines)


def _get_param_names_from_key(key: str):
    """从 2-pair key 中提取两个参数名."""
    left = key.split(" × ")[0]
    right = key.split(" × ")[1]
    return left.split(".")[0], right.split(".")[0]


def format_markdown_report_grouped(
    result: CoverageResult,
    domain: OperatorAttributeDomain,
    case_count: int,
    title: str = "Coverage Report",
) -> str:
    """生成按算子参数/非算子参数分组的 Markdown 覆盖率报告."""
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **算子**: {domain.operator_name}")
    lines.append(f"- **用例数**: {case_count}")
    lines.append("")

    # 按 is_operator_param 分组
    op_params = {n: p for n, p in domain.params.items() if p.is_operator_param}
    non_op_params = {n: p for n, p in domain.params.items() if not p.is_operator_param}

    lines.append(f"- **算子参数数**: {len(op_params)}")
    lines.append(f"- **非算子参数数**: {len(non_op_params)}")
    lines.append("")

    # ------------------------------------------------
    # 1-pair: 算子参数
    # ------------------------------------------------
    lines.append("## 一、1-pair 覆盖")

    for group_name, param_dict in [("算子参数", op_params), ("非算子参数", non_op_params)]:
        lines.append(f"### {group_name}")
        lines.append("")

        rows = []
        group_rates = []
        for param_name in sorted(param_dict.keys()):
            param_result = result.one_pair.get(param_name, {})
            param_type = domain.params[param_name].constraint_type if param_name in domain.params else ""
            for attr_name in sorted(param_result.keys()):
                info = param_result[attr_name]
                rows.append((param_name, param_type, attr_name, info["count"], info["total"], info["rate"]))
                group_rates.append(info["rate"])

        if rows:
            lines.append("| 参数 | 类型 | 属性 | 已覆盖 | 可能取值 | 覆盖率 |")
            lines.append("|------|------|------|-------:|--------:|------:|")
            for p, pt, a, c, t, r in rows:
                lines.append(f"| {p} | {pt} | {a} | {c} | {t} | {_pct(r)} |")
            lines.append("")
            avg = sum(group_rates) / len(group_rates) if group_rates else 0.0
            lines.append(f"**{group_name}平均 1-pair 覆盖率**: {_pct(avg)}")
        else:
            lines.append("_无可用数据_")
        lines.append("")

    # ------------------------------------------------
    # 2-pair: 分组统计
    # ------------------------------------------------
    lines.append("## 二、2-pair 覆盖")

    # 定义分组规则
    def _pair_group(key: str):
        n1, n2 = _get_param_names_from_key(key)
        if n1 in op_params and n2 in op_params:
            return "op_op"
        if (n1 in op_params and n2 in non_op_params) or (n1 in non_op_params and n2 in op_params):
            return "op_non"
        return "non_non"

    pair_groups = {
        "算子参数 × 算子参数": "op_op",
        "算子参数 × 非算子参数": "op_non",
        "非算子参数 × 非算子参数": "non_non",
    }

    for group_name, group_tag in pair_groups.items():
        group_keys = sorted(k for k in result.two_pair.keys() if _pair_group(k) == group_tag)
        if not group_keys:
            continue

        lines.append(f"### {group_name}")
        lines.append("")

        rows = []
        group_rates = []
        for key in group_keys:
            info = result.two_pair[key]
            rows.append((key, info["count"], info["total"], info["rate"]))
            group_rates.append(info["rate"])

        lines.append("| 序号 | 属性对 | 已覆盖 | 可能组合 | 覆盖率 |")
        lines.append("|----:|--------|-------:|--------:|------:|")
        for i, (k, c, t, r) in enumerate(rows, 1):
            lines.append(f"| {i} | {k} | {c} | {t} | {_pct(r)} |")
        lines.append("")
        avg = sum(group_rates) / len(group_rates) if group_rates else 0.0
        lines.append(f"**{group_name}平均 2-pair 覆盖率**: {_pct(avg)}")

        uncovered = [(k, c, t) for k, c, t, r in rows if r == 0.0]
        if uncovered:
                lines.append("")
                lines.append(f"**未覆盖的组合 (共 {len(uncovered)} 个):**")
                lines.append("")
                lines.append("| 序号 | 属性对 |")
                lines.append("|----:|--------|")
                for i, (k, _, _) in enumerate(uncovered, 1):
                    lines.append(f"| {i} | {k} |")
        lines.append("")

    # ------------------------------------------------
    # 跨属性 2-pair: 分组统计
    # ------------------------------------------------
    if result.two_pair_cross:
        lines.append("## 三、跨属性 2-pair 覆盖（不同属性跨参数配对）")
        lines.append("")

        for group_name, group_tag in pair_groups.items():
            group_keys = sorted(k for k in result.two_pair_cross.keys() if _pair_group(k) == group_tag)
            if not group_keys:
                continue

            lines.append(f"### {group_name}")
            lines.append("")

            rows = []
            group_rates = []
            for key in group_keys:
                info = result.two_pair_cross[key]
                rows.append((key, info["count"], info["total"], info["rate"]))
                group_rates.append(info["rate"])

            lines.append("| 序号 | 属性对 | 已覆盖 | 可能组合 | 覆盖率 |")
            lines.append("|----:|--------|-------:|--------:|------:|")
            for i, (k, c, t, r) in enumerate(rows, 1):
                lines.append(f"| {i} | {k} | {c} | {t} | {_pct(r)} |")
            lines.append("")
            avg = sum(group_rates) / len(group_rates) if group_rates else 0.0
            lines.append(f"**{group_name}跨属性平均 2-pair 覆盖率**: {_pct(avg)}")

            uncovered = [(k, c, t) for k, c, t, r in rows if r == 0.0]
            if uncovered:
                lines.append("")
                lines.append(f"**未覆盖的组合 (共 {len(uncovered)} 个):**")
                lines.append("")
                lines.append("| 序号 | 属性对 |")
                lines.append("|----:|--------|")
                for i, (k, _, _) in enumerate(uncovered, 1):
                    lines.append(f"| {i} | {k} |")
            lines.append("")

    return "\n".join(lines)
