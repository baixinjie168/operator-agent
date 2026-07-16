import ctypes
import torch
import atk.tasks.backends.lib_interface.acl_wrapper as acl_wrapper
import torch.nn.functional as F
import re
from atk.common.log import Logger
from atk.tasks.backends.lib_interface.acl_wrapper import Int64, Uint64, AclTensorStruct, TORCH_TO_ACLTYPE, nnopbase, AclFormat, aclnn, AclnnStatus, TensorPtr ,AclTensorlistStruct
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import *

logging = Logger().get_logger()


@register("aclnn_function")
class AclnnReflectionPad1dBackward(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """
        初始化输入参数并整合算子输出到输入参数列表

        处理流程：
        1. 转换输入参数为aclnn所需的c++格式
        2. 收集算子输出元数据
        3. 将算子输出张量地址追加到输入参数

        核心参数：
        1. input_args：算子的入参列表，应严格符合算子的c++函数接口原型的参数顺序和类型，均应转换为ctypes式或AclTensorStruct(List[ctypes | AclTensorStruct])
        2. output_packages：算子的出参数据包列表，用于精度对比在调用算子后解析数据包以获取算子输出(List[AclTensorStruct])
        3. tensor数据包数据结构：
            class AclTensorStruct:
                data: AclTensor       # aclTensor的c++对象（真正传递给算子的直接参数）
                addr: int             # 内存地址（用于PyTorch转换时的指针操作）
                shape: List[int]      # 张量形状
                dtype: AclDataType    # 数据类型
        """
        input_tmp={}
        input_args = []  # 算子的入参列表
        output_packages = []  # 算子的出参数据包列表

        # 获取到算子参数的入参
        param_list = self.get_param_names_excluding_last_two(self.get_cpp_func_signature_type())
        # 获取到算子参数的类型
        param_type = self.parse_operator_params(self.get_cpp_func_signature_type())

        self.handle_special_param(self.get_cpp_func_signature_type(), input_data)

        self.handle_attr_param(input_tmp, param_list)

        # === 处理输入参数 ===
        # 将输入数据转换为aclnn所需的c++格式
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            if name in input_tmp:
                continue
            data = self.backend.convert_input_data(kwarg, name=name)
            if name in param_list:
                input_tmp[name] = data

        # 构造算子调用的入参顺序
        for i, arg_name in enumerate(param_list):
            data = input_tmp.get(arg_name)
            if data is not None:
                if isinstance(data, list):
                    input_args.extend(data)
                else:
                    input_args.append(data)
            else:
                # 没有传入的参数设置为nullptr
                type = param_type.get(arg_name)
                if type == "aclTensor":
                    AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
                    null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
                    null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)  # 把这个空指针类型转换为tensor指针类型
                    input_args.append(null_tensor_ptr)
                elif type == "aclTensorList":
                    AclTensorListPtr = ctypes.POINTER(AclTensorList)
                    null_tensorlist_ptr = AclTensorListPtr()  # 默认为 None
                    input_args.append(null_tensorlist_ptr)
                else:
                    input_args.append(ctypes.c_void_p(None))

        # === 处理标杆输出 ===
        # 收集算子输出，并储存根据输出中的shape和dtype信息生成的AclTensorStruct数据结构
        # 输出数据结构说明：
        output_list = self.output.split(',')
        for index, output in enumerate(output_list):
            data = input_tmp.get(output)
            if isinstance(data, list):
                output_packages.append(data[0])
            else:
                output_packages.append(data)
        return input_args, output_packages

    def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
        if name is not None:
            found = self.get_config_by_name(self.task_result.case_config.inputs, name)
            if not isinstance(found, list) and found.shape is not None:
                return torch.Size(found.shape)
        return None

    def get_format(self, input_data: InputDataset, index=None, name=None):
        found = None
        if name is not None:
            found = self.get_config_by_name(self.task_result.case_config.inputs, name)
        if found is not None and not isinstance(found, list):
            format = found.format
            return self.get_acl_format(format)
        return AclFormat.ACL_FORMAT_ND

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnReflectionPad1dBackwardGetWorkspaceSize( const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)"
    def get_param_names_excluding_last_two(self, func_str):
        """
        返回参数名称列表。
        如果函数签名包含 "GetWorkspaceSize"（二段式算子），则排除最后两个参数（workspaceSize 和 executor）；
        否则返回全部参数名。
        """
        # 提取参数列表
        match = re.search(r'\(([^)]*)\)', func_str)
        if not match:
            return []

        params_str = match.group(1)

        # 提取所有参数名
        param_names = []
        for param in params_str.split(','):
            param = param.strip()
            # 提取最后的标识符作为参数名（忽略 const、* 等修饰）
            name_match = re.search(r'\*?\s*(\w+)\s*$', param)
            if name_match:
                param_names.append(name_match.group(1))

        # 判断是否为二段式调用（包含 GetWorkspaceSize）
        if "GetWorkspaceSize" in func_str:
            # 排除最后两个（workspaceSize 和 executor）
            return param_names[:-2] if len(param_names) >= 2 else param_names
        else:
            # 非二段式，返回全部参数名
            return param_names


    def handle_attr_param(self, input_tmp, param_list):
        for config in self.task_result.case_config.inputs:
            if not isinstance(config, list):
                if config.name not in param_list:
                    continue
                if config.type == "attr":
                    range_val = config.range_values[0] if isinstance(config.range_values, list) else config.range_values
                    ctype = self.get_ctype(config.dtype)
                    if ctype == ctypes.c_char_p:  # Python的字符串是Unicode字符串，C风格的字符串一般是字节字符串（如ASCII或UTF-8）
                        range_val = range_val.encode('utf-8')
                    input_tmp[config.name] = [ctype(range_val)]
            else:
                data = []
                data_name = config[0].name
                if data_name not in param_list:
                    continue
                for config_item in config:
                    if config_item.type == "attrs" or config_item.type == "attr_tuple":
                        range_val = config_item.range_values[0] if isinstance(config_item.range_values, list) else config_item.range_values
                        data.append(self.get_ctype(config_item.dtype)(range_val))
                        input_tmp[data_name] = nnopbase.create_x_list(data)

    def handle_special_param(self, operator_name, input_data):

        if "aclnnBatchMatMulWeightNz" in operator_name:
            found_mat2_transposed = next((config for config in self.task_result.case_config.inputs if config.name == "mat2_transposed"), None)
            # 转置的情况
            if found_mat2_transposed.range_values:
                # 转换
                input_data.kwargs['mat2'] = input_data.kwargs['mat2'].permute(0, 2, 1, 4, 3).reshape(
                    input_data.kwargs['mat2'].shape[0],  # b
                    input_data.kwargs['mat2'].shape[1] * input_data.kwargs['mat2'].shape[4],  # k1 * k0 = k
                    input_data.kwargs['mat2'].shape[2] * input_data.kwargs['mat2'].shape[3]  # n1 * n0 = n
                )
                input_data.kwargs['self'] = input_data.kwargs['self'].permute(0, 2, 1)
                return
            input_data.kwargs['mat2'] = input_data.kwargs['mat2'].reshape(
                input_data.kwargs['mat2'].shape[0],  # a
                input_data.kwargs['mat2'].shape[2] * input_data.kwargs['mat2'].shape[3],  # c*d
                input_data.kwargs['mat2'].shape[1] * input_data.kwargs['mat2'].shape[4]  # b*e
            )


    def parse_operator_params(self, func_signature: str):
        """
        从函数签名中解析出参数名和参数类型的字典（不包含const和*）

        Args:
            func_signature: 函数签名字符串

        Returns:
            字典，key为参数名，value为参数类型（不含const和*）
        """
        # 提取函数括号内的所有参数
        params_match = re.search(r'\((.*?)\)\s*$', func_signature, re.DOTALL)
        if not params_match:
            return {}

        params_str = params_match.group(1)

        # 按逗号分割参数，但要注意嵌套的尖括号
        params = []
        current_param = ""
        angle_bracket_count = 0
        paren_count = 0

        for char in params_str:
            if char == '<':
                angle_bracket_count += 1
            elif char == '>':
                angle_bracket_count -= 1
            elif char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and angle_bracket_count == 0 and paren_count == 0:
                params.append(current_param.strip())
                current_param = ""
                continue

            current_param += char

        if current_param.strip():
            params.append(current_param.strip())

        # 解析每个参数
        result = {}
        for param in params:
            # 匹配参数类型和参数名
            match = re.match(r'^(const\s+)?(.+?)\s+(\*?\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*=\s*.*)?$', param.strip())
            if match:
                param_type = match.group(2).strip()
                param_name = match.group(4).strip()

                # 去掉类型中的const和*（包括指针标记）
                # 先去掉const
                param_type = param_type.replace('const ', '').replace('const', '').strip()
                # 再去掉所有*
                param_type = param_type.replace('*', '').strip()

                result[param_name] = param_type

        return result

    def get_config_by_name(self, configs, target_name: str):
        """
        根据name获取配置数据

        Args:
            configs: 混合数组，元素可能是InputCaseConfig对象或List[InputCaseConfig]
            target_name: 目标name值

        Returns:
            找到的配置对象或列表，未找到返回None
        """
        for item in configs:
            if isinstance(item, list):
                # 如果是列表，检查第一个元素的name
                if item and hasattr(item[0], 'name') and item[0].name == target_name:
                    return item  # 返回整个列表
            elif hasattr(item, 'name') and item.name == target_name:
                return item  # 返回单个对象

        return None

    def get_acl_format(self, format_str):
        FORMAT_MAPPING = {
            'ND': AclFormat.ACL_FORMAT_ND,
            'NZ': AclFormat.ACL_FORMAT_FRACTAL_NZ,
            'NCHW': AclFormat.ACL_FORMAT_NCHW,
            'NC': AclFormat.ACL_FORMAT_NC,
            'HWCN': AclFormat.ACL_FORMAT_HWCN,
            'NHWC': AclFormat.ACL_FORMAT_NHWC,
            'NC1HWC0': AclFormat.ACL_FORMAT_NC1HWC0,
            'NCL': AclFormat.ACL_FORMAT_NCL,
            'NCDHW': AclFormat.ACL_FORMAT_NCDHW,
            'NDHWC': AclFormat.ACL_FORMAT_NDHWC,
        }

        if format_str in FORMAT_MAPPING:
            return FORMAT_MAPPING[format_str]
        else:
            return AclFormat.ACL_FORMAT_ND

    def get_ctype(self, type_str):
        PYTYPE_TO_CTYPE = {
            "float": ctypes.c_float,
            "float32": ctypes.c_float,
            "double": ctypes.c_double,
            "int": ctypes.c_int64,
            "int8_t": ctypes.c_int8,
            "int8": ctypes.c_int8,
            "int32_t": ctypes.c_int32,
            "int32": ctypes.c_int32,
            "int64_t": ctypes.c_int64,
            "int64": ctypes.c_int64,
            "uint8_t": ctypes.c_uint8,
            "uint8": ctypes.c_uint8,
            "uint32_t": ctypes.c_uint32,
            "uint32": ctypes.c_uint32,
            "uint64_t": ctypes.c_uint64,
            "uint64": ctypes.c_uint64,
            "bool": ctypes.c_bool,
            "attr_bool": ctypes.c_bool,
            "str": ctypes.c_char_p,
            "string": ctypes.c_char_p,
        }

        if type_str in PYTYPE_TO_CTYPE:
            return PYTYPE_TO_CTYPE[type_str]
        else:
            raise ValueError(f"Unsupported CTYPE format: {type_str}")


@register("function")
class Function(BaseApi):
    """Auto-generated CPU reference class for aclnnReflectionPad1dBackward."""

    _OP_NAME = "aclnnReflectionPad1dBackward"
    _SIG_STR = """aclnnStatus aclnnReflectionPad1dBackwardGetWorkspaceSize( const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)"""
    _INPUT_PARAM_NAMES = ['gradOutput', 'self', 'padding']

    def calculate_by_torch(self, gradOutput, self_t, padding_raw):
        x = self_t.clone().detach().requires_grad_(True)

        # 创建 ReflectionPad1d 层
        pad_layer = torch.nn.ReflectionPad1d(padding_raw)

        # 执行前向传播
        y = pad_layer(x)

        gradInput, = torch.autograd.grad(y, x, grad_outputs=gradOutput)
        return gradInput

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # Build name->value mapping from input_data.args using case_config.inputs names.
        # ATK base_dataset.py appends each generated value to input_data.args in JSON inputs[] order.
        # case_config.inputs[i].name gives the parameter name for args[i].
        _param_map = {}
        if input_data.args and hasattr(self, 'task_result') and self.task_result.case_config.inputs:
            flat_configs = self.task_result.case_config.flatten_list(self.task_result.case_config.inputs)
            for idx, conf in enumerate(flat_configs):
                if idx < len(input_data.args) and conf.name:
                    _param_map[conf.name] = input_data.args[idx]
                    # Also index by lowercase for case-insensitive lookup
                    _param_map[conf.name.lower()] = input_data.args[idx]
        # Fallback: also index kwargs if present (for non-ACLNN paths)
        if input_data.kwargs:
            for k, v in input_data.kwargs.items():
                if v is not None:
                    _param_map[k] = v
                    _param_map[k.lower()] = v

        def _get_param(name, default=None):
            v = _param_map.get(name)
            if v is None:
                v = _param_map.get(name.lower())
            if v is not None:
                return v
            return default

        # Tensor validation: ensures a param is actually a torch.Tensor
        # ATK may sometimes generate non-tensor values (e.g. int) for params
        # expected to be tensors, causing AttributeError during computation
        def _get_tensor(name, default=None):
            v = _get_param(name, default)
            return v if isinstance(v, torch.Tensor) else default

        # ACLNN operator: aclnnReflectionPad1dBackward
        # C++ signature: aclnnStatus aclnnReflectionPad1dBackwardGetWorkspaceSize( const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)

        # --- Input parameters from signature ---
        # gradOutput           (const aclTensor*      )  # input tensor
        # self                 (const aclTensor*      )  # input tensor
        # padding              (const aclIntArray*    )  # attr
        # gradInput            (aclTensor*            )  # output tensor

        # ACLNN aclnnReflectionPad1dBackward CPU golden:
        # Compute the gradient of ReflectionPad1d given gradOutput and self.
        # Forward: F.pad(self, [pad_left, pad_right], mode='reflect')
        # Backward: accumulate gradOutput at each contributing input position.
        #
        # Forward index mapping for self length L:
        #   output[pad_left + j]              = self[j]                               (direct, j in [0, L))
        #   output[pad_left - j]              = self[j]   for j in [1, pad_left]     (left reflection)
        #   output[2L + pad_left - 2 - j]     = self[j]   for j in [L-1-pad_right, L-1)  (right reflection)
        # Backward: reverse the mapping by slicing and summing.

        # Extract inputs
        gradOutput = _get_tensor("gradOutput")
        self_t = _get_tensor("self")

        # Padding (const aclIntArray*, length 2).
        # _get_param may return only one value for duplicate-name group params.
        input_tmp = {}
        for config in self.task_result.case_config.inputs:
            if not isinstance(config, list):
                if config.type == "attr":
                    range_val = config.range_values[0] if isinstance(config.range_values, list) else config.range_values
                    input_tmp[config.name] = [range_val]
            else:
                data = []
                data_name = config[0].name
                for config_item in config:
                    if config_item.type == "attrs" or config_item.type == "attr_tuple":
                        range_val = config_item.range_values[0] if isinstance(config_item.range_values, list) else config_item.range_values
                        data.append(range_val)
                        input_tmp[data_name] = data

        padding_raw = input_tmp["padding"]
        return self.calculate_by_torch(gradOutput, self_t, padding_raw)