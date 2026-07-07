# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2026/1/5 19:44
功能：参数约束关系实现
"""
import ast
import re
from collections import defaultdict
from typing import List, Dict

import z3
from pydantic import BaseModel

from agent.generators.atk_common_utils.case_config import CaseConfig
from agent.generators.operator_param_models.case_generate import CaseGenerate
from agent.generators.param_constraint_solve.customize_expression_solver_utils import CustomizeConstraintPatch
from agent.generators.param_constraint_solve.z3_expression_solver_utils import Z3ConstraintBuilder, ExpressionPreprocessor, ASTtoZ3Converter
from agent.generators.common_model_definition import InterParamConstraint, InterConstraintsRuleType, OperatorRule
from agent.generators.common_utils.common_dispatcher import CommonDispatcher
from agent.generators.common_utils.data_handle_utils import DataHandleUtil
from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import ParamModelConfig, DataMatchMap
from agent.generators.data_definition.param_models_def import ParameterPropertyData, ParamRangeValueType

logger = LazyLogger()


class ParamSetValueFlag(BaseModel):
    dtype: bool = False
    shape: bool = False
    range_values: bool = False
    format: bool = False
    length: bool = False


class ParamConstraintUtils(CommonDispatcher):
    def __init__(self, case: CaseConfig, case_generate_instance: CaseGenerate,
                 inter_param_constraints: List[InterParamConstraint], operator_rule_data: OperatorRule,
                 param_combinations: Dict[str, ParameterPropertyData] = None,
                 is_generate_real_data: bool = False):
        self.inter_param_constraints = inter_param_constraints
        # 形如{'x1': {'DType': 'BFLOAT16', 'DataProfile': 'NaN', 'DimCount': '2', 'DimProperty': 'Has_Large_Size',
        # 'DataProfile': 'SubNormal', 'Memory': 'Contiguous'}, 'epsilon': {'Value': '1e-5'},
        # 'additionalOutput': {'Mode': 'True'}}
        self.param_combinations = param_combinations
        self.operator_rule_data = operator_rule_data
        # 是否生成真实数据，若为FALSE只保留生成数据模型的名称，用于后续实际执行的时候调用生成真实数据
        self.is_generate_real_data = is_generate_real_data
        # 数据生成实例，用于在不满足约束条件时，重新生成参数的数据
        self.case_generate_instance = case_generate_instance
        self.operator_name = case.name
        self.case = case
        self.case_input_map = {case_input.name: case_input for case_input in case.inputs}
        self.broadcast_master_params = defaultdict(List[InterParamConstraint])
        self.has_set_value_param = defaultdict(ParamSetValueFlag)
        self.relation_params = list(operator_rule_data.inputs.keys())
        self.customize_constraint_patch = CustomizeConstraintPatch(case=case,
                                                                   case_generate_instance=case_generate_instance,
                                                                   inter_param_constraints=inter_param_constraints,
                                                                   operator_rule_data=operator_rule_data,
                                                                   param_combinations=param_combinations,
                                                                   is_generate_real_data=is_generate_real_data)
        self.dtype_domain_data, self.format_domain_data = self.get_param_domain_value()

    def get_param_domain_value(self) -> tuple[Dict[str, List], Dict[str, List]]:
        """
        获取所有参数的dtype和format的值域，用于构建求解条件
        :return: 每个参数dtype的值域数据和format值域数据
        """
        dtype_domain_data = {}
        format_domain_data = {}
        for param_name in self.case_input_map.keys():
            dtype_domain = self.generate_dtype_string_domain(param_name)
            format_domain = self.generate_format_string_domain(param_name)
            if dtype_domain:
                dtype_domain_data[param_name] = dtype_domain
            if format_domain:
                format_domain_data[param_name] = format_domain
        return dtype_domain_data, format_domain_data

    def is_param_all_input(self, relation_params: List[str]):
        """
        判断参数的角色是否都为输入
        :param relation_params: 约束相关的参数名称
        """
        for param_name in relation_params:
            if param_name not in self.case_input_map:
                logger.error(f"Can't match this parameter in input params, param name : {param_name}")
                return False
        return True

    def set_has_value_param_status(self, param_name, constraint_type: str):
        """
        将已确定的参数的属性的状态在self.has_set_value_param中设置为True，
        :param param_name: 参数名称
        :param constraint_type: 约束的类型
        :return: None
        """
        param_set_value_flag = self.has_set_value_param.get(param_name, ParamSetValueFlag())
        for type_key, type_list in DataMatchMap.CONSTRAINT_TYPE_MAP.items():
            if constraint_type in type_list:
                param_set_value_flag.__setattr__(type_key, True)
        self.has_set_value_param[param_name] = param_set_value_flag

    def correct_operator_param(self):
        """
        修复case的参数，在源数据上修改
        :return: None
        """
        logger.info(f"Start correct case param, operator name : {self.operator_name}")
        customize_constraints = []
        z3_constraints = []

        for constraint_relation in self.inter_param_constraints:
            relation_type = constraint_relation.expr_type
            if relation_type in ParamModelConfig.STRICT_CONSTRAINT_TYPE:
                customize_constraints.append(constraint_relation)
            else:
                # 这里用于筛选输出参数，如果约束表达式中包含输出参数，此表达式不进行处理，但有些情况下，表达式中除了关于输出参数的表达式，
                # 还可能有只包含输入参数的子表达式，如果不处理可能会导致部分表达式不生效
                # if not self.is_param_all_input(constraint_relation.relation_params):
                #     continue
                logger.debug(f"Relation type : {relation_type}, expr : {constraint_relation.expr}, use z3 solver")
                z3_constraints.append(constraint_relation)
        if not self.solve_z3_constraints(z3_constraints):
            logger.error(
                f"Correct case param failed because Z3 constraints are "
                f"invalid or unsatisfied, operator name : {self.operator_name}"
            )
            return False
        for customize_constraint in customize_constraints:
            relation_type = customize_constraint.expr_type
            logger.debug(f"Relation type : {relation_type}, use strict constraint logical")
            strict_check_result, relation_params = self.customize_constraint_patch.dispatch(relation_type,
                                                                                            customize_constraint)
            if not strict_check_result:
                logger.debug(
                    f"Relation type : {relation_type}, use strict constraint logical failed, "
                    f"check result : {strict_check_result}")
                return False
            for param_name in relation_params:
                self.set_has_value_param_status(param_name, relation_type)
            logger.debug(f"Relation type : {relation_type}, use strict constraint logical success")
        logger.info(f"End correct case param, operator name : {self.operator_name}")
        return True

    def generate_dtype_string_domain(self, param_name: str) -> List[str]:
        """
        获取数据类型dtype可取值，用列表表示
        :param param_name: 参数名称
        :return: List[dtype]
        """
        param_attribute = self.operator_rule_data.inputs.get(param_name)
        if param_attribute is None:
            param_attribute = self.operator_rule_data.outputs.get(param_name)
        if param_attribute is None:
            logger.error(f"Param : {param_name}, dtype domain is None")
            return []
        dtype_domain, _ = DataHandleUtil.get_relevant_attribute_value(param_name, param_attribute.dtype, "dtype")
        if dtype_domain is None:
            logger.error(f"Param : {param_name}, dtype domain is not relevant")
            return []
        dtype_domain = [DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.get(dtype) for dtype in dtype_domain]
        return dtype_domain

    def generate_format_string_domain(self, param_name: str) -> List[str]:
        """
        获取数据格式format的可取值，用列表表示
        :param param_name: 参数名称
        :return: List[format]
        """
        param_attribute = self.operator_rule_data.inputs.get(param_name)
        if param_attribute is None:
            param_attribute = self.operator_rule_data.outputs.get(param_name)
        if param_attribute is None:
            logger.warning(f"Param : {param_name}, format domain is None")
            return []
        format_domain, _ = DataHandleUtil.get_relevant_attribute_value(param_name, param_attribute.format, "format")
        if format_domain is None:
            logger.warning(f"Param : '{param_name}', format domain is not relevant")
            return []
        return format_domain

    @staticmethod
    def adapter_dtype_in_expr(expr: str) -> str:
        """
        适配表达式中的数据类型，如原始表达式中为INT4, 统一替换为int, INT8统一替换为int8,
        需要替换的值位于DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP
        :param expr: 需要被替换的表达式
        :return: 替换之后的表达式
        """
        if not isinstance(expr, str):
            logger.info(f"Expr type is not string, can't be replace, expr : {expr}")
            return expr
        for key, value in DataMatchMap.ACL_DTYPE_TRANSFER_TENSOR_MAP.items():
            if isinstance(expr, str):
                expr = re.sub(rf"\b{re.escape(key)}\b", value, expr)
        return expr

    def choice_no_conflicts_expr(self, builder: Z3ConstraintBuilder, param_union_expr: List[str],
                                 param_static_expr_list: List[str]) -> None:
        """
        筛选和参数静态表达式不冲突的初始值表达式。
        使用 Z3 push/pop 增量检测，避免重复创建求解器。同时将确认不冲突的表达式永久添加至 builder。

        :param builder: 已声明变量并添加了 JSON 约束的 Z3ConstraintBuilder 实例
        :param param_union_expr: 关联表达式列表（会被原地修改）
        :param param_static_expr_list: 参数静态表达式列表
        """
        if not param_static_expr_list:
            return

        # 快速路径：一次添加全部，若 SAT 直接返回
        builder.solver.push()
        for static_expr in param_static_expr_list:
            replace_expr = ExpressionPreprocessor.apply_keyword_replace(static_expr)
            if ExpressionPreprocessor.validate_expression(replace_expr):
                tree = ast.parse(replace_expr, mode='eval')
                converter = ASTtoZ3Converter(builder)
                z3_constraint = converter.visit(tree.body)
                if z3_constraint is not None:
                    builder.solver.assert_and_track(z3_constraint, f"batch_chk:{static_expr[:50]}")
        if builder.solver.check() == z3.sat:
            builder.solver.pop()
            # 永久添加所有静态表达式
            for static_expr in param_static_expr_list:
                param_union_expr.append(static_expr)
                replace_expr = ExpressionPreprocessor.apply_keyword_replace(static_expr)
                if ExpressionPreprocessor.validate_expression(replace_expr):
                    tree = ast.parse(replace_expr, mode='eval')
                    converter = ASTtoZ3Converter(builder)
                    z3_constraint = converter.visit(tree.body)
                    if z3_constraint is not None:
                        builder.solver.assert_and_track(z3_constraint, f"perm:{static_expr[:50]}")
            logger.debug(f"Batch check SAT, all {len(param_static_expr_list)} exprs kept")
            return
        builder.solver.pop()

        # 回退路径：逐个 push/pop 增量检测
        for static_expr in param_static_expr_list:
            builder.solver.push()
            try:
                replace_expr = ExpressionPreprocessor.apply_keyword_replace(static_expr)
                is_sat = False
                if ExpressionPreprocessor.validate_expression(replace_expr):
                    tree = ast.parse(replace_expr, mode='eval')
                    converter = ASTtoZ3Converter(builder)
                    z3_constraint = converter.visit(tree.body)
                    if z3_constraint is not None:
                        builder.solver.assert_and_track(z3_constraint, f"chk:{static_expr[:50]}")
                        if builder.solver.check() == z3.sat:
                            is_sat = True
            finally:
                builder.solver.pop()

            if is_sat:
                logger.debug(f"Check param value expr, result : no conflicts, expr : {static_expr}")
                param_union_expr.append(static_expr)
                # 永久添加
                replace_expr = ExpressionPreprocessor.apply_keyword_replace(static_expr)
                if ExpressionPreprocessor.validate_expression(replace_expr):
                    tree = ast.parse(replace_expr, mode='eval')
                    converter = ASTtoZ3Converter(builder)
                    z3_constraint = converter.visit(tree.body)
                    if z3_constraint is not None:
                        builder.solver.assert_and_track(z3_constraint, f"perm:{static_expr[:50]}")

    def build_param_dtype_constraint(self, constraint_exprs: List[str],
                                     builder: Z3ConstraintBuilder, check: bool = True) -> None:
        """
        查找是否有参数的dtype属性取值已确定，如果有，是否和已有的规则冲突，如不冲突，则添加为求解条件，避免每次求解结构都相同，如冲突，则不添加
        :param constraint_exprs: 约束表达式对象（会被原地修改）
        :param builder: Z3求解器构建器
        :param check: 是否立即执行冲突检测
        """
        static_value_exprs = []
        relation_param = list(self.case_input_map.keys())
        for param in relation_param:
            static_value_exprs.append(f"{param}.dtype == '{self.case_input_map.get(param).dtype}'")
        if check:
            self.choice_no_conflicts_expr(builder=builder, param_union_expr=constraint_exprs,
                                          param_static_expr_list=static_value_exprs)
        else:
            constraint_exprs.extend(static_value_exprs)

    def build_param_range_value_constraint(self, constraint_exprs: List[str],
                                           builder: Z3ConstraintBuilder, check: bool = True) -> None:
        """
        对于range_value，如果range_value是浮点数或整数，则直接加入求解条件表达式，否则不加入求解表达式
        :param constraint_exprs: 约束条件数据（会被原地修改）
        :param builder: Z3求解器构建器
        :param check: 是否立即执行冲突检测
        """
        static_range_value_expr_list = []
        relation_params = list(self.case_input_map.keys())
        for param_name in relation_params:
            param_attr = self.operator_rule_data.inputs.get(param_name)
            if param_attr is None:
                param_attr = self.operator_rule_data.outputs.get(param_name)
            if param_attr is None:
                continue
            param_range_value_constraint, value_type = DataHandleUtil.get_relevant_attribute_value(
                param_name, param_attr.allowed_range_value, "allowed_range_value")
            if param_range_value_constraint is None:
                continue
            param_range_value_expr_list = []
            for value_rule in param_range_value_constraint:
                if value_type == ParamRangeValueType.ENUM.value:
                    if value_rule is None:
                        param_range_value_expr_list.append(f"{param_name} is {value_rule}")
                    elif isinstance(value_rule, str):
                        param_range_value_expr_list.append(f"{param_name}.range_value == '{value_rule}'")
                    elif isinstance(value_rule, list):
                        value_rule_expr_list = []
                        for val_index, val in enumerate(value_rule):
                            value_rule_expr_list.append(f"{param_name}[{val_index}] == {val}")
                        value_rule_expr_str = " and ".join(value_rule_expr_list)
                        value_rule_expr_str = f"({value_rule_expr_str})"
                        param_range_value_expr_list.append(value_rule_expr_str)
                    else:
                        param_range_value_expr_list.append(f"{param_name}.range_value == {value_rule}")
                else:
                    if len(value_rule) >= 2:
                        param_range_value_expr_list.append(
                            f"({param_name}.range_value > {value_rule[0]} and {param_name}.range_value < {value_rule[1]})")
                    else:
                        logger.error(
                            f"Param name : {param_name}, allowed range value is invalid, type : 'range', value : '{value_rule}'")
            param_range_value_expr = " or ".join(param_range_value_expr_list)
            static_range_value_expr_list.append(param_range_value_expr)
        if check:
            self.choice_no_conflicts_expr(builder=builder, param_union_expr=constraint_exprs,
                                          param_static_expr_list=static_range_value_expr_list)
        else:
            constraint_exprs.extend(static_range_value_expr_list)

    def build_param_shape_len_constraint(self, constraint_exprs: List[str], builder: Z3ConstraintBuilder,
                                         check: bool = True) -> None:
        """
        基于约束数据冲的dimension创建shape的len条件
        :param constraint_exprs: 约束表达式对象（会被原地修改）
        :param builder: Z3求解器构建器
        :param check: 是否立即执行冲突检测
        """
        shape_len_static_value_expr_list = []
        relation_params = list(self.case_input_map.keys())
        for param_name in relation_params:
            param_ori_data = self.operator_rule_data.inputs.get(
                param_name) if param_name in self.operator_rule_data.inputs else self.operator_rule_data.outputs.get(
                param_name)
            if param_ori_data is None:
                continue
            param_shape_dimension, _ = DataHandleUtil.get_relevant_attribute_value(param_name,
                                                                                   param_ori_data.dimensions,
                                                                                   "dimensions")
            # 如果参数的shape属性为None，表明该参数不是tensor，没有shape属性，不需要添加shape约束
            if param_shape_dimension is None:
                continue
            if isinstance(param_shape_dimension, list):
                param_shape_dimension = list(set(param_shape_dimension))
                shape_len_static_value_expr_list.append(f"len({param_name}.shape) in {param_shape_dimension}")
            elif isinstance(param_shape_dimension, int):
                shape_len_static_value_expr_list.append(f"len({param_name}.shape) == {param_shape_dimension}")
            else:
                logger.warning(f"Can't get valid tensor dimension data, param name : {param_name}")

        if check:
            self.choice_no_conflicts_expr(builder=builder, param_union_expr=constraint_exprs,
                                          param_static_expr_list=shape_len_static_value_expr_list)
        else:
            constraint_exprs.extend(shape_len_static_value_expr_list)

    def build_param_shape_constraint(self, constraint_exprs: List[str],
                                     builder: Z3ConstraintBuilder, check: bool = True) -> None:
        """
        对于shape,如果该参数的parameter_constraints中shape约束没有axis_value的约束表达式，则不添加确定值的约束，避免约束冲突，
        此时将axis_value中的约束表达式添加至列表中，如果没有axis_value的约束表达式，则将shape当前取值添加至表达式列表
        :param constraint_exprs: 约束表达式对象（会被原地修改）
        :param builder: Z3求解器构建器
        :param check: 是否立即执行冲突检测
        """
        shape_static_value_expr_list = []
        relation_params = list(self.case_input_map.keys())
        for param_name in relation_params:
            if param_name not in self.case_input_map:
                continue
            case_param_shape = self.case_input_map.get(param_name).shape
            # 如果参数的shape属性为None，表明该参数不是tensor，没有shape属性，不需要添加shape约束
            if case_param_shape is None:
                continue
            shape_static_value_expr_list.append(f"{param_name}.shape == {case_param_shape}")

        if check:
            self.choice_no_conflicts_expr(builder=builder, param_union_expr=constraint_exprs,
                                          param_static_expr_list=shape_static_value_expr_list)
        else:
            constraint_exprs.extend(shape_static_value_expr_list)

    def build_param_format_constraint(self, constraint_exprs: List[str],
                                      builder: Z3ConstraintBuilder, check: bool = True) -> None:
        """
        查找是否有参数的format属性取值已确定，如果有，是否和已有的规则冲突，如不冲突，则添加为求解条件，避免每次求解结构都相同，如冲突，则不添加
        :param constraint_exprs: 约束表达式对象（会被原地修改）
        :param builder: Z3求解器构建器
        :param check: 是否立即执行冲突检测
        """
        format_static_value_expr_list = []
        relation_param = list(self.case_input_map.keys())
        for param in relation_param:
            if self.case_input_map.get(param).format is not None:
                format_static_value_expr_list.append(f"{param}.format == '{self.case_input_map.get(param).format}'")
        if check:
            self.choice_no_conflicts_expr(builder=builder, param_union_expr=constraint_exprs,
                                          param_static_expr_list=format_static_value_expr_list)
        else:
            constraint_exprs.extend(format_static_value_expr_list)

    def build_param_length_constraint(self, constraint_exprs: List[str],
                                      builder: Z3ConstraintBuilder, check: bool = True) -> None:
        length_static_value_expr_list = []
        relation_param = list(self.case_input_map.keys())
        for param in relation_param:
            if self.case_input_map.get(param).type in ParamModelConfig.LIST_ATK_TYPE and self.case_input_map.get(
                    param).length is not None:
                length_static_value_expr_list.append(f"len({param}) == {self.case_input_map.get(param).length}")
        if check:
            self.choice_no_conflicts_expr(builder=builder, param_union_expr=constraint_exprs,
                                          param_static_expr_list=length_static_value_expr_list)
        else:
            constraint_exprs.extend(length_static_value_expr_list)

    def declare_param_in_z3(self, builder: Z3ConstraintBuilder, is_print_log=False):
        """
        声明每个变量，指定变量的type(Tensor, scalar，list)，以及数据类型(float, int, string, bool)
        :param is_print_log: 日志中是否打印详细信息
        :param builder: z3求解器实例

        :return: None
        """
        for param_name in self.case_input_map.keys():
            param_info = self.case_input_map.get(param_name)
            if not param_info:
                continue
            param_type = param_info.type
            z3_param_type = DataMatchMap.Z3_VAR_TYPE_MAP.get(param_type, "tensor")
            param_dtype = param_info.dtype
            param_length = param_info.length
            range_values = self.case_input_map.get(param_name).range_values
            if param_type in ParamModelConfig.TENSOR_ATK_TYPE:
                dtype_domain = self.dtype_domain_data.get(param_name)
                format_domain = self.format_domain_data.get(param_name)
                builder.declare_var(param_name, type_hint=z3_param_type, dtype=param_dtype, allowed_dtypes=dtype_domain,
                                    allowed_formats=format_domain, range_value=range_values,
                                    length=param_length if z3_param_type == "tensor_list" else None,
                                    is_print_log=is_print_log)
            else:
                builder.declare_var(param_name, z3_param_type, dtype=param_dtype, range_value=range_values,
                                    length=param_length, is_print_log=is_print_log)

        standalone_none_params = set()
        for constraint in self.inter_param_constraints:
            expr_text = constraint.expr
            for param_name in self.case_input_map:
                # 检测参数名是否出现在任何 is None / is not None 上下文中
                if re.search(rf'\b{re.escape(param_name)}\s+is\s+(?:not\s+)?None\b', expr_text):
                    standalone_none_params.add(param_name)

        for param_name in self.case_input_map:
            param_combination_data = self.param_combinations.get(param_name)
            if param_combination_data is None:
                continue
            is_optional = param_combination_data.is_optional
            if is_optional:
                continue
            builder.solver.add(builder.var_map[param_name].is_present)

    def solve_z3_constraints(self, z3_constraints: List[InterParamConstraint]):
        """
        针对所有的参数关联表达式，使用z3求解器求解
        :param: z3_constraints: 表达式列表，包含所有需要求解的表达式
        """
        expr_list = [constraint_expr.expr for constraint_expr in z3_constraints if
                     constraint_expr.expr is not None and constraint_expr.expr]
        logger.info(f"Start solving solution of constraints by Z3, operator name : {self.operator_name}")
        builder = Z3ConstraintBuilder()
        self.declare_param_in_z3(builder=builder, is_print_log=True)

        # 先添加 JSON 约束到求解器
        json_expr_dict = {f"json:{expr[:50]}": expr for expr in expr_list}
        builder.add_constraints(expr_str_dict=json_expr_dict)

        # 收集所有静态表达式，一次性批量冲突检测（5 次 choice_no_conflicts_expr → 1 次）
        all_static = []
        self.build_param_dtype_constraint(all_static, builder, check=False)
        self.build_param_format_constraint(all_static, builder, check=False)
        self.build_param_shape_constraint(all_static, builder, check=False)
        self.build_param_range_value_constraint(all_static, builder, check=False)
        self.build_param_shape_len_constraint(all_static, builder, check=False)
        self.build_param_length_constraint(all_static, builder, check=False)

        if all_static:
            self.choice_no_conflicts_expr(builder=builder, param_union_expr=expr_list,
                                          param_static_expr_list=all_static)

        logger.info("Start whole expr solve")
        solver_result = builder.solve()
        if not solver_result:
            logger.error(
                f"Z3 solver error, no solution can satisfy constraints, operator name : {self.operator_name}")
            return False
        logger.info(f"End solving solution of constraints by Z3, operator name : {self.operator_name}")

        absent_params = [param_name for param_name in self.case_input_map.keys()
                         if solver_result.get(param_name, {}).get('is_present') is False]
        for param_name in absent_params:
            self.case.inputs = [inp for inp in self.case.inputs if inp.name not in absent_params]
            self.case_input_map.pop(param_name)

        for param_name in self.case_input_map.keys():
            property_dict = solver_result.get(param_name, {})
            for field_name in ParamSetValueFlag.model_fields.keys():
                # 参数必须在case_input_map中，且数据值不为None，如非tensor参数没有shape属性，但是Z3求解器会对表达式涉及到的所有参数所有属性都会求解
                param_attr_ori_status = param_name in self.case_input_map and getattr(
                    self.case_input_map.get(param_name), field_name, None) is not None
                attr_value = property_dict.get(field_name, None)
                if param_attr_ori_status and attr_value is not None:
                    self.case_input_map[param_name].__setattr__(field_name, attr_value)
        return True
