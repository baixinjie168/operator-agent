import ctypes
import random

import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
from functools import reduce
from operator import mul

import atk.tasks.backends.lib_interface.acl_wrapper as acl_wrapper

from atk.tasks.backends.lib_interface.acl_wrapper import (
    AclTensor, AclnnStatus, OpExecutor, Uint64, Int64, VoidPtr,
    nnopbase, ascendcl, aclnn, AclTensorlistStruct,
    AclDataType, AclFormat, TensorPtr, AclTensorStruct,
    TORCH_TO_ACLTYPE, ACLTYPE_TO_CTYPE
)
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("function")
class MethodNpuFormatCast(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_tmp = {}
        # 获取到算子参数的入参
        param_list = ["srcTensor", "dstFormat", "additionalDtype"]
        self.handle_attr_param(input_tmp, param_list)
        input_data.kwargs["dstFormat"] = input_tmp["dstFormat"][0].value
        input_data.kwargs["additionalDtype"] = input_tmp["additionalDtype"][0].value

        found_srcTensor = self.get_config_by_name(self.task_result.case_config.inputs, "srcTensor")
        if found_srcTensor.format == 'NZ' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_ND:
            return self.format_nz_to_nd(input_data, with_output)
        if found_srcTensor.format == 'NHWC' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NC1HWC0:
            return self.format_nhwc_to_nc1hwc0(input_data, with_output)
        if found_srcTensor.format == 'ND' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_FRACTAL_NZ:
            return self.format_nd_to_nz(input_data, with_output)
        if found_srcTensor.format == 'NDHWC' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NDC1HWC0:
            return self.format_ndhwc_to_ndc1hwc0(input_data, with_output)
        if found_srcTensor.format == 'NDC1HWC0' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NDHWC:
            return self.format_ndc1hwc0_to_ndhwc(input_data, with_output)
        if found_srcTensor.format == 'NDC1HWC0' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NCDHW:
            return self.format_ndc1hwc0_to_ncdhw(input_data, with_output)
        if found_srcTensor.format == 'NCHW' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NC1HWC0:
            return self.format_nchw_to_nc1hwc0(input_data, with_output)
        if found_srcTensor.format == 'NCHW' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_FRACTAL_Z:
            return self.format_nchw_to_fz(input_data, with_output)
        if found_srcTensor.format == 'NCDHW' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NDC1HWC0:
            return self.format_ncdhw_to_ndc1hwc0(input_data, with_output)
        if found_srcTensor.format == 'NCDHW' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FRACTAL_Z_3D:
            return self.format_ncdhw_to_fz3d(input_data, with_output)
        if found_srcTensor.format == 'NC1HWC0' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NHWC:
            return self.format_nc1hwc0_to_nhwc(input_data, with_output)
        if found_srcTensor.format == 'NC1HWC0' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NCHW:
            return self.format_nc1hwc0_to_nchw(input_data, with_output)
        if found_srcTensor.format == 'HWCN' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_FRACTAL_Z:
            return self.format_hwcn_to_fz(input_data, with_output)
        if found_srcTensor.format == 'FRACTAL_Z' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NCHW:
            return self.format_fz_to_nchw(input_data, with_output)
        if found_srcTensor.format == 'FRACTAL_Z' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_HWCN:
            return self.format_fz_to_hwcn(input_data, with_output)
        if found_srcTensor.format == 'FRACTAL_Z_3D' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NCDHW:
            return self.format_fz3d_to_ncdhw(input_data, with_output)


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

    def get_config_by_name(self, configs, target_name: str):
        for item in configs:
            if isinstance(item, list):
                # 如果是列表，检查第一个元素的name
                if item and hasattr(item[0], 'name') and item[0].name == target_name:
                    return item  # 返回整个列表
            elif hasattr(item, 'name') and item.name == target_name:
                return item  # 返回单个对象

        return None

    # ===== 工具函数 =====
    def _ceil_div(self, x: int, y: int) -> int:
        return (x + y - 1) // y

    def _pad_len(self, x: int, align: int) -> int:
        return self._ceil_div(x, align) * align - x

    def _c0_of_torch_dtype(self, td: torch.dtype) -> int:
        if td in (torch.int8, torch.uint8):                     return 32
        if td in (torch.int16, torch.float16, torch.bfloat16):  return 16
        if td in (torch.int32, torch.float32, torch.uint32):                  return 8
        if td == torch.int64:                                   return 4
        raise ValueError(f"暂不支持的 dtype: {td}")

    def format_nz_to_nd(self, input_data: InputDataset, with_output: bool = False):
        # ===== 读取入参 =====
        src_t = input_data.kwargs["srcTensor"]                # 任意 ND 形状（视为 ND）
        dst_format = input_data.kwargs.get("dstFormat", 2)    # 目标 ND=2
        additional_dtype = input_data.kwargs["additionalDtype"]

        # ===== 把任意 ND 归一到 (H, N, C) =====
        in_shape = tuple(int(d) for d in src_t.shape)
        in_len = len(in_shape)
        if in_len == 1:
            axis_h, axis_n, axis_c = 1, 1, in_shape[0]
        elif in_len == 2:
            axis_h, axis_n, axis_c = 1, in_shape[0], in_shape[1]
        else:
            axis_h = 1 if in_len <= 2 else int(torch.tensor(in_shape[:-2]).prod().item())
            axis_n = in_shape[-2]
            axis_c = in_shape[-1]

        C0 = self._c0_of_torch_dtype(src_t.dtype)
        NI = 16
        C1 = self._ceil_div(axis_c, C0)
        NO = self._ceil_div(axis_n, NI)
        n_pad = self._pad_len(axis_n, NI)
        c_pad = self._pad_len(axis_c, C0)

        # ========= Step A: ND -> NZ（用于确定补齐与打包；全用 torch，支持 bf16）=========
        x = src_t.detach().to("cpu").contiguous().view(axis_h, axis_n, axis_c)
        if n_pad or c_pad:
            x = F.pad(x, (0, c_pad, 0, n_pad, 0, 0))  # (C右, N右) 只右侧补零

        # (H,N,C) -> (H, NO, NI, C1, C0) -> (H, C1, NO, NI, C0)  [NZ 5D]
        nz5 = x.view(axis_h, NO, NI, C1, C0).permute(0, 3, 1, 2, 4).contiguous()

        # ========= Step B: NZ -> ND（按你的 golden 的逆变换）=========
        # golden:
        # tmp = reshape(H, C1, NO, NI, C0) -> transpose(0,2,3,1,4) -> reshape(H, NO*NI, C1*C0) -> crop[:N,:C]
        tmp = nz5.reshape(axis_h, C1, NO, NI, C0)
        tmp = tmp.permute(0, 2, 3, 1, 4).contiguous()           # (H, NO, NI, C1, C0)
        tmp = tmp.reshape(axis_h, NO * NI, C1 * C0)             # (H, N_pad, C_pad)
        out_hnc = tmp[:, :axis_n, :axis_c].contiguous()         # 去 pad，(H, N, C)

        # 还原到原始 ND 形状
        if in_len == 1:
            out = out_hnc.view(axis_c)
        elif in_len == 2:
            out = out_hnc.view(axis_n, axis_c)
        else:
            out = out_hnc.view(*in_shape)

        return out.to(dtype=src_t.dtype, copy=False)

    def format_nhwc_to_nc1hwc0(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 3)  # 目标 NCDHW=30
        additional_dtype = input_data.kwargs["additionalDtype"]

        # ---- 参数与对齐 ----
        N, H, W, C = map(int, src_t.shape)
        C0 = self._c0_of_torch_dtype(src_t.dtype)
        if (src_t.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        C1 = self._ceil_div(C, C0)
        c_pad = self._pad_len(C, C0)

        x = src_t.detach().to("cpu").contiguous()
        if c_pad:
            x = F.pad(x, (0, c_pad, 0, 0, 0, 0, 0, 0))  # (Wl,Wr, Hl,Hr, Cl,Cr, Nl,Nr)

        # (N, C, H, W) -> (NO, NI, C1, C0, H, W) -> (C1, H, W, NO, NI, C0)  [FZ3D 6D]
        x = x.view(N, H, W, C1, C0)
        NC1HWC0 = x.permute(0, 3, 1, 2, 4).contiguous()

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return NC1HWC0.to(dtype=src_t.dtype, copy=False)

    def format_nd_to_nz(self, input_data: InputDataset, with_output: bool = False):
        # ===== 读取入参 =====
        src_t = input_data.kwargs["srcTensor"]  # torch.Tensor, 任意 ND 形状
        dst_format = input_data.kwargs.get("dstFormat", 29)  # 期望 NZ=29
        additional_dtype = input_data.kwargs["additionalDtype"]

        # ===== 逻辑：把任意 ND 归一化为 (H, N, C) → pad → (H, NO, NI, C1, C0) → (H, C1, NO, NI, C0) =====
        in_shape = tuple(int(d) for d in src_t.shape)
        in_len = len(in_shape)

        # 计算 H, N, C
        if in_len == 1:
            axis_h, axis_n, axis_c = 1, 1, in_shape[0]
        elif in_len == 2:
            axis_h, axis_n, axis_c = 1, in_shape[0], in_shape[1]  # 注意：原伪码这里写成了 == 是个笔误
        else:
            axis_h = reduce(lambda x, y: x * y, in_shape[:-2])
            axis_n = in_shape[-2]
            axis_c = in_shape[-1]

        axis_c0 = self._c0_of_torch_dtype(src_t.dtype)  # C0 由 dtype 的 32B 对齐推导
        axis_ni = 16  # NI 固定为 16

        axis_c1 = self._ceil_div(axis_c, axis_c0)
        axis_no = self._ceil_div(axis_n, axis_ni)
        c_pad = self._pad_len(axis_c, axis_c0)
        n_pad = self._pad_len(axis_n, axis_ni)

        # 把 src 转为 (H, N, C) 的 numpy；BF16 先转 FP32 以兼容 numpy
        if src_t.dtype == torch.bfloat16:
            x_np = src_t.detach().to("cpu").to(torch.float32).contiguous().view(axis_h, axis_n, axis_c).numpy()
            _final_dtype = torch.bfloat16
        else:
            x_np = src_t.detach().to("cpu").contiguous().view(axis_h, axis_n, axis_c).numpy()
            _final_dtype = src_t.dtype

        # 只在 N/C 右侧补 0
        x_np = np.pad(
            x_np,
            pad_width=((0, 0), (0, n_pad), (0, c_pad)),
            mode="constant",
            constant_values=(0, 0),
        )

        # (H, N, C) → (H, NO, NI, C1, C0)
        x_np = x_np.reshape(axis_h, axis_no, axis_ni, axis_c1, axis_c0)

        # 置换到 NZ 目标顺序：(H, C1, NO, NI, C0)
        out_np = np.transpose(x_np, axes=(0, 3, 1, 2, 4))

        # 转回 torch，保持与输入一致的 dtype（BF16 走回转）
        out_t = torch.from_numpy(out_np).to(dtype=_final_dtype, copy=False)
        return out_t

    def format_ndhwc_to_ndc1hwc0(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NDHWC
        dst_format = input_data.kwargs.get("dstFormat", 32)  # 目标 NDC1HWC0=32
        additional_dtype = input_data.kwargs["additionalDtype"]

        # ---- 参数与对齐 ----
        N, D, H, W, C = map(int, src_t.shape)
        C0 = self._c0_of_torch_dtype(src_t.dtype)
        if (src_t.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        C1 = self._ceil_div(C, C0)
        c_pad = self._pad_len(C, C0)

        x = src_t.detach().to("cpu").contiguous()
        if c_pad:
            x = F.pad(x, (0, c_pad, 0, 0, 0, 0, 0, 0, 0, 0))  # (Wl,Wr, Hl,Hr, DL,DR, Cl,Cr, Nl,Nr)

        # (N, D, H, W, C) -> (N, D, H, W, C1, C0) [NDC1HWC0 6D]
        x = x.view(N, D, H, W, C1, C0)
        NDC1HWC0 = x.permute(0, 1, 4, 2, 3, 5).contiguous()

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return NDC1HWC0.to(dtype=src_t.dtype, copy=False)

    def format_ndc1hwc0_to_ndhwc(self, input_data: InputDataset, with_output: bool = False):
        # 入参
        src = input_data.kwargs["srcTensor"]  # 形状: (N, C, D, H, W) NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 27)  # 期望输出是 NCDHW=30
        _ = input_data.kwargs["additionalDtype"]

        def _ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y

        def _c0_of_torch_dtype(td: torch.dtype) -> int:
            # 32B 对齐
            if td in (torch.int8, torch.uint8):                     return 32  # 1B
            if td in (torch.int16, torch.float16, torch.bfloat16):  return 16  # 2B
            if td in (torch.int32, torch.float32, torch.uint32):                  return 8  # 4B
            if td == torch.int64:                                   return 4  # 8B
            raise ValueError(f"暂不支持的 dtype: {td}")

        # === 第一步：NCDHW -> NDC1HWC0（在 CPU，用 torch 实现）===
        N, D, H, W, C = map(int, src.shape)
        C0 = _c0_of_torch_dtype(src.dtype)
        C1 = _ceil_div(C, C0)
        c_pad = C1 * C0 - C

        x = src.detach().to("cpu").contiguous()  # 支持 bf16，无需转 numpy
        if c_pad > 0:
            # pad 格式：(W_l, W_r, H_l, H_r, D_l, D_r, C_l, C_r, N_l, N_r)
            x = F.pad(x, (c_pad, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        # (N,C,D,H,W) -> (N,C1,C0,D,H,W) -> (N,D,C1,H,W,C0)  (NDC1HWC0)
        x_ndc1hwc0 = x.view(N, D, H, W, C1, C0).permute(0, 1, 4, 2, 3, 5).contiguous()

        # === 第二步：NDC1HWC0 -> NCDHW（按你给的标杆公式，仍用 torch）===
        # in:  (N, D, C1, H, W, C0)
        tmp = x_ndc1hwc0.reshape(N, D, C1, H, W, C0)
        tmp = tmp.permute(0, 1, 3, 4, 2, 5).contiguous()  # (N, C1, C0, D, H, W)
        tmp = tmp.reshape(N, D, H, W, C1 * C0)  # (N, C1*C0, D, H, W)
        out = tmp[:, :, :, :, :C].contiguous()  # 去 padding -> (N, C, D, H, W)

        # baseline 返回期望输出（NCDHW，dtype不变，CPU）
        return out.to(dtype=src.dtype, copy=False)

    def format_ndc1hwc0_to_ncdhw(self, input_data: InputDataset, with_output: bool = False):
        # 入参
        src = input_data.kwargs["srcTensor"]  # 形状: (N, C, D, H, W) NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 30)  # 期望输出是 NCDHW=30
        _ = input_data.kwargs["additionalDtype"]

        # === 第一步：NCDHW -> NDC1HWC0（在 CPU，用 torch 实现）===
        N, C, D, H, W = map(int, src.shape)
        C0 = self._c0_of_torch_dtype(src.dtype)
        C1 = self._ceil_div(C, C0)
        c_pad = C1 * C0 - C

        x = src.detach().to("cpu").contiguous()  # 支持 bf16，无需转 numpy
        if c_pad > 0:
            # pad 格式：(W_l, W_r, H_l, H_r, D_l, D_r, C_l, C_r, N_l, N_r)
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, c_pad, 0, 0))
        # (N,C,D,H,W) -> (N,C1,C0,D,H,W) -> (N,D,C1,H,W,C0)  (NDC1HWC0)
        x_ndc1hwc0 = x.view(N, C1, C0, D, H, W).permute(0, 3, 1, 4, 5, 2).contiguous()

        # === 第二步：NDC1HWC0 -> NCDHW（按你给的标杆公式，仍用 torch）===
        # in:  (N, D, C1, H, W, C0)
        tmp = x_ndc1hwc0.reshape(N, D, C1, H, W, C0)
        tmp = tmp.permute(0, 2, 5, 1, 3, 4).contiguous()  # (N, C1, C0, D, H, W)
        tmp = tmp.reshape(N, C1 * C0, D, H, W)  # (N, C1*C0, D, H, W)
        out = tmp[:, :C, :, :, :].contiguous()  # 去 padding -> (N, C, D, H, W)

        # baseline 返回期望输出（NCDHW，dtype不变，CPU）
        return out.to(dtype=src.dtype, copy=False)

    def format_nchw_to_nc1hwc0(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 3)  # 目标 NCDHW=30
        additional_dtype = input_data.kwargs["additionalDtype"]

        # ---- 参数与对齐 ----
        N, C, H, W = map(int, src_t.shape)
        C0 = self._c0_of_torch_dtype(src_t.dtype)
        if (src_t.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        C1 = self._ceil_div(C, C0)
        c_pad = self._pad_len(C, C0)

        x = src_t.detach().to("cpu").contiguous()
        if c_pad:
            x = F.pad(x, (0, 0, 0, 0, 0, c_pad,))  # (Wl,Wr, Hl,Hr, Cl,Cr, Nl,Nr)

        # (N, C, H, W) -> (NO, NI, C1, C0, H, W) -> (C1, H, W, NO, NI, C0)  [FZ3D 6D]
        x = x.view(N, C1, C0, H, W)
        NC1HWC0 = x.permute(0, 1, 3, 4, 2).contiguous()

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return NC1HWC0.to(dtype=src_t.dtype, copy=False)

    def format_nchw_to_fz(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 4)  # 目标 NCDHW=30
        additional_dtype = input_data.kwargs["additionalDtype"]
        # ---- 参数与对齐 ----
        N, C, H, W = map(int, src_t.shape)
        C0 = self._c0_of_torch_dtype(src_t.dtype)
        if (src_t.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        NI = 16
        C1 = self._ceil_div(C, C0)
        NO = self._ceil_div(N, NI)
        n_pad = self._pad_len(N, NI)
        c_pad = self._pad_len(C, C0)

        x = src_t.detach().to("cpu").contiguous()
        if n_pad or c_pad:
            x = F.pad(x, (0, 0, 0, 0, 0, c_pad, 0, n_pad))  # (Wl,Wr, Hl,Hr, Cl,Cr, Nl,Nr)

        # (N, C, H, W) -> (NO, NI, C1, C0, H, W) -> (C1, H, W, NO, NI, C0)  [FZ3D 6D]
        x = x.view(NO, NI, C1, C0, H, W)
        z_6d = x.permute(2, 4, 5, 0, 1, 3).contiguous()

        z_4d = z_6d.reshape(C1 * H * W, NO, NI, C0)

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return z_4d.to(dtype=src_t.dtype, copy=False)

    def format_ncdhw_to_ndc1hwc0(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 32)  # 目标 NDC1HWC0=32
        additional_dtype = input_data.kwargs["additionalDtype"]
        # ---- 参数与对齐 ----
        N, C, D, H, W = map(int, src_t.shape)
        C0 = self._c0_of_torch_dtype(src_t.dtype)
        if (src_t.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        C1 = self._ceil_div(C, C0)
        c_pad = self._pad_len(C, C0)

        x = src_t.detach().to("cpu").contiguous()
        if c_pad:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, c_pad, 0, 0))  # (Wl,Wr, Hl,Hr, DL,DR, Cl,Cr, Nl,Nr)

        # (N, C, D, H, W) -> (N, C1, C0, D, H, W) -> (C1, H, W, NO, NI, C0)  [FZ3D 6D]
        x = x.view(N, C1, C0, D, H, W)
        NDC1HWC0 = x.permute(0, 3, 1, 4, 5, 2).contiguous()

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return NDC1HWC0.to(dtype=src_t.dtype, copy=False)

    def format_ncdhw_to_fz3d(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 32)  # 目标 NDC1HWC0=32
        additional_dtype = input_data.kwargs["additionalDtype"]

        # ---- 参数与对齐 ----
        N, C, D, H, W = map(int, src_t.shape)

        C0 = self._c0_of_torch_dtype(src_t.dtype)
        if (src_t.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        NI = 16
        C1 = self._ceil_div(C, C0)
        NO = self._ceil_div(N, NI)
        c_pad = self._pad_len(C, C0)
        n_pad = self._pad_len(N, NI)

        x = src_t.detach().to("cpu").contiguous()
        if n_pad or c_pad:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, c_pad, 0, n_pad))  # (Wl,Wr, Hl,Hr, DL,DR, Cl,Cr, Nl,Nr)

        # (N, C, D, H, W) -> (NO, NI, C1, C0, D, H, W) -> (D, C1, H, W, NO, NI, C0)  [FZ3D]
        x = x.view(NO, NI, C1, C0, D, H, W)
        z3D_7d = x.permute(4, 2, 5, 6, 0, 1, 3).contiguous()
        z3D_4d = z3D_7d.reshape(D * C1 * H * W, NO, NI, C0)

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return z3D_4d.to(dtype=src_t.dtype, copy=False)

    def format_nc1hwc0_to_nhwc(self, input_data: InputDataset, with_output: bool = False):

        # 入参
        src = input_data.kwargs["srcTensor"]  # 形状: (N, C, H, W) NCHW
        dst_format = input_data.kwargs.get("dstFormat", 1)  # 期望输出是 NCDHW=30
        _ = input_data.kwargs["additionalDtype"]

        # === 第一步：NCDHW -> NDC1HWC0（在 CPU，用 torch 实现）===
        N, H, W, C = map(int, src.shape)
        C0 = self._c0_of_torch_dtype(src.dtype)
        C1 = self._ceil_div(C, C0)
        c_pad = C1 * C0 - C

        x = src.detach().to("cpu").contiguous()  # 支持 bf16，无需转 numpy
        if c_pad > 0:
            # pad 格式：(W_l, W_r, H_l, H_r, D_l, D_r, C_l, C_r, N_l, N_r)
            x = F.pad(x, (c_pad, 0, 0, 0, 0, 0, 0, 0))
        # (N,C,D,H,W) -> (N,C1,C0,D,H,W) -> (N,D,C1,H,W,C0)  (NDC1HWC0)
        x_nc1hwc0 = x.view(N, H, W, C1, C0).permute(0, 3, 1, 2, 4).contiguous()

        # === 第二步：NDC1HWC0 -> NCDHW（按你给的标杆公式，仍用 torch）===
        # in:  (N, D, C1, H, W, C0)
        tmp = x_nc1hwc0.reshape(N, C1, H, W, C0)
        tmp = tmp.permute(0, 2, 3, 1, 4).contiguous()  # (N, C1, C0, D, H, W)
        tmp = tmp.reshape(N, H, W, C1 * C0)  # (N, C1*C0, H, W)
        out = tmp[:, :, :, :C].contiguous()  # 去 padding -> (N, C, H, W)

        # baseline 返回期望输出（NCDHW，dtype不变，CPU）
        return out.to(dtype=src.dtype, copy=False)

    def format_nc1hwc0_to_nchw(self, input_data: InputDataset, with_output: bool = False):
        # 入参
        src = input_data.kwargs["srcTensor"]  # 形状: (N, C, H, W) NCHW
        dst_format = input_data.kwargs.get("dstFormat", 0)  # 期望输出是 NCDHW=30
        _ = input_data.kwargs["additionalDtype"]

        # === 第一步：NCDHW -> NDC1HWC0（在 CPU，用 torch 实现）===
        N, C, H, W = map(int, src.shape)
        C0 = self._c0_of_torch_dtype(src.dtype)
        C1 = self._ceil_div(C, C0)
        c_pad = C1 * C0 - C

        x = src.detach().to("cpu").contiguous()  # 支持 bf16，无需转 numpy
        if c_pad > 0:
            # pad 格式：(W_l, W_r, H_l, H_r, D_l, D_r, C_l, C_r, N_l, N_r)
            x = F.pad(x, (0, 0, 0, 0, 0, c_pad, 0, 0))
        # (N,C,D,H,W) -> (N,C1,C0,D,H,W) -> (N,D,C1,H,W,C0)  (NDC1HWC0)
        x_nc1hwc0 = x.view(N, C1, C0, H, W).permute(0, 1, 3, 4, 2).contiguous()

        # === 第二步：NDC1HWC0 -> NCDHW（按你给的标杆公式，仍用 torch）===
        # in:  (N, D, C1, H, W, C0)
        tmp = x_nc1hwc0.reshape(N, C1, H, W, C0)
        tmp = tmp.permute(0, 1, 4, 2, 3).contiguous()  # (N, C1, C0, D, H, W)
        tmp = tmp.reshape(N, C1 * C0, H, W)  # (N, C1*C0, H, W)
        out = tmp[:, :C, :, :].contiguous()  # 去 padding -> (N, C, H, W)

        # baseline 返回期望输出（NCDHW，dtype不变，CPU）
        return out.to(dtype=src.dtype, copy=False)

    def format_hwcn_to_fz(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 4)  # 目标 NCDHW=30
        additional_dtype = input_data.kwargs["additionalDtype"]
        # ---- 参数与对齐 ----
        H, W, C, N = map(int, src_t.shape)
        C0 = self._c0_of_torch_dtype(src_t.dtype)
        if (src_t.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        NI = 16
        C1 = self._ceil_div(C, C0)
        NO = self._ceil_div(N, NI)
        n_pad = self._pad_len(N, NI)
        c_pad = self._pad_len(C, C0)

        x = src_t.detach().to("cpu").contiguous()
        if n_pad or c_pad:
            x = F.pad(x, (0, n_pad, 0, c_pad, 0, 0, 0, 0))  # (Wl,Wr, Hl,Hr, Cl,Cr, Nl,Nr)

        # (H, W, C, N) -> (H, W, C1, C0, NO, NI) -> (C1, H, W, NO, NI, C0)  [FZ3D 6D]
        x = x.view(H, W, C1, C0, NO, NI)
        z_6d = x.permute(2, 0, 1, 4, 5, 3).contiguous()

        z_4d = z_6d.reshape(C1 * H * W, NO, NI, C0)

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return z_4d.to(dtype=src_t.dtype, copy=False)

    def format_fz_to_nchw(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 0)  # 目标 NCDHW=30
        additional_dtype = input_data.kwargs["additionalDtype"]

        # ---- 参数与对齐 ----
        N, C, H, W = map(int, src_t.shape)
        C0 = self._c0_of_torch_dtype(src_t.dtype)
        if (src_t.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        NI = 16
        C1 = self._ceil_div(C, C0)
        NO = self._ceil_div(N, NI)
        n_pad = self._pad_len(N, NI)
        c_pad = self._pad_len(C, C0)

        x = src_t.detach().to("cpu").contiguous()
        if n_pad or c_pad:
            x = F.pad(x, (0, 0, 0, 0, 0, c_pad, 0, n_pad))  # (Wl,Wr, Hl,Hr, Cl,Cr, Nl,Nr)

        # (N, C, H, W) -> (NO, NI, C1, C0, H, W) -> (C1, H, W, NO, NI, C0)  [FZ3D 6D]
        x = x.view(NO, NI, C1, C0, H, W)
        z_6d = x.permute(2, 4, 5, 0, 1, 3).contiguous()

        z_4d = z_6d.reshape(C1 * H * W, NO, NI, C0)
        print(z_4d.shape)
        tmp = z_4d.reshape(C1, H, W, NO, NI, C0)
        tmp = tmp.permute(3, 4, 0, 5, 1, 2).contiguous()
        tmp = tmp.reshape(NO * NI, C1 * C0, H, W)
        out = tmp[:N, :C, :, :].contiguous()

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return out.to(dtype=src_t.dtype, copy=False)

    def format_fz_to_hwcn(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 16)  # 目标 NCDHW=30
        additional_dtype = input_data.kwargs["additionalDtype"]
        # ---- 参数与对齐 ----
        H, W, C, N = map(int, src_t.shape)
        C0 = self._c0_of_torch_dtype(src_t.dtype)
        if (src_t.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        NI = 16
        C1 = self._ceil_div(C, C0)
        NO = self._ceil_div(N, NI)
        n_pad = self._pad_len(N, NI)
        c_pad = self._pad_len(C, C0)

        x = src_t.detach().to("cpu").contiguous()
        if n_pad or c_pad:
            x = F.pad(x, (0, n_pad, 0, c_pad, 0, 0, 0, 0))  # (Wl,Wr, Hl,Hr, Cl,Cr, Nl,Nr)

        # (N, C, H, W) -> (NO, NI, C1, C0, H, W) -> (C1, H, W, NO, NI, C0)  [FZ3D 6D]
        x = x.view(H, W, C1, C0, NO, NI)
        z_6d = x.permute(2, 0, 1, 4, 5, 3).contiguous()

        z_4d = z_6d.reshape(C1 * H * W, NO, NI, C0)
        print(z_4d.shape)
        tmp = z_4d.reshape(C1, H, W, NO, NI, C0)
        tmp = tmp.permute(1, 2, 0, 5, 3, 4).contiguous()
        tmp = tmp.reshape(H, W, C1 * C0, NO * NI)
        out = tmp[:, :, :C, :N].contiguous()

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return out.to(dtype=src_t.dtype, copy=False)

    def format_fz3d_to_ncdhw(self, input_data: InputDataset, with_output: bool = False):

        src_t = input_data.kwargs["srcTensor"]  # 逻辑 NCDHW
        dst_format = input_data.kwargs.get("dstFormat", 30)  # 目标 NCDHW=30
        additional_dtype = input_data.kwargs["additionalDtype"]

        # ---- 参数与对齐 ----
        N, C, D, H, W = map(int, src_t.shape)
        C0 = self._c0_of_torch_dtype(src_t.dtype)
        NI = 16
        C1 = self._ceil_div(C, C0)
        NO = self._ceil_div(N, NI)
        n_pad = self._pad_len(N, NI)
        c_pad = self._pad_len(C, C0)

        # ========= Step A: 先把 NCDHW“打包”为 FZ3D（7D），再展平为 4D 存储 =========
        # pad N/C（右侧）
        x = src_t.detach().to("cpu").contiguous()
        if n_pad or c_pad:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, c_pad, 0, n_pad))  # (Wl,Wr, Hl,Hr, Dl,Dr, Cl,Cr, Nl,Nr)

        # (N, C, D, H, W) -> (NO, NI, C1, C0, D, H, W) -> (D, C1, H, W, NO, NI, C0)  [FZ3D 7D]
        x = x.view(NO, NI, C1, C0, D, H, W)
        z3d_7d = x.permute(4, 2, 5, 6, 0, 1, 3).contiguous()

        # 按 golden 的输入约定，FZ3D 的 storage 采用 4D： (DC1HW, NO, NI, C0)
        z3d_4d = z3d_7d.reshape(D * C1 * H * W, NO, NI, C0)

        # ========= Step B: 按 golden 执行 FZ3D(4D) -> NCDHW =========
        # axis_dc1hw = D*C1*H*W；axis_no=NO；axis_ni=NI；axis_c0=C0
        # dst_shape = (N, C, D, H, W)
        tmp = z3d_4d.reshape(D, C1, H, W, NO, NI, C0)  # (D, C1, H, W, NO, NI, C0)
        tmp = tmp.permute(4, 5, 1, 6, 0, 2, 3).contiguous()  # (NO, NI, C1, C0, D, H, W)
        tmp = tmp.reshape(NO * NI, C1 * C0, D, H, W)  # (N_pad, C_pad, D, H, W)
        out = tmp[:N, :C, :, :, :].contiguous()  # 去掉 pad，回到 (N, C, D, H, W)

        # baseline：返回 CPU、与输入同 dtype（torch 原生支持 bf16）
        return out.to(dtype=src_t.dtype, copy=False)

@register("aclnn_function")
class AclnnNpuFormatCast(AclnnBaseApi):
    def _ceil_div(self, x: int, y: int) -> int:
        return (x + y - 1) // y

    def _pad_len(self, x: int, align: int) -> int:
        return self._ceil_div(x, align) * align - x

    def _c0_of_torch_dtype(self, td: torch.dtype) -> int:
        if td in (torch.int8, torch.uint8):                     return 32
        if td in (torch.int16, torch.float16, torch.bfloat16):  return 16
        if td in (torch.int32, torch.float32, torch.uint32):                  return 8
        if td == torch.int64:                                   return 4
        raise ValueError(f"暂不支持的 dtype: {td}")

    def _contiguous_strides(self, shape_tuple):
        s = [1] * len(shape_tuple)
        for i in range(len(shape_tuple) - 2, -1, -1):
            s[i] = s[i + 1] * int(shape_tuple[i + 1])
        return s

    def init_by_input_data(self, input_data: InputDataset):
        input_tmp = {}
        # 获取到算子参数的入参
        param_list = ["srcTensor", "dstFormat", "additionalDtype"]
        self.handle_attr_param(input_tmp, param_list)
        input_data.kwargs["dstFormat"] = input_tmp["dstFormat"][0].value
        input_data.kwargs["additionalDtype"] = input_tmp["additionalDtype"][0].value

        found_srcTensor = self.get_config_by_name(self.task_result.case_config.inputs, "srcTensor")
        if found_srcTensor.format == 'NZ' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_ND:
            return self.format_nz_to_nd(input_data)
        if found_srcTensor.format == 'NHWC' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NC1HWC0:
            return self.format_nhwc_to_nc1hwc0(input_data)
        if found_srcTensor.format == 'ND' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_FRACTAL_NZ:
            return self.format_nd_to_nz(input_data)
        if found_srcTensor.format == 'NDHWC' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NDC1HWC0:
            return self.format_ndhwc_to_ndc1hwc0(input_data)
        if found_srcTensor.format == 'NDC1HWC0' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NDHWC:
            return self.format_ndc1hwc0_to_ndhwc(input_data)
        if found_srcTensor.format == 'NDC1HWC0' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NCDHW:
            return self.format_ndc1hwc0_to_ncdhw(input_data)
        if found_srcTensor.format == 'NCHW' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NC1HWC0:
            return self.format_nchw_to_nc1hwc0(input_data)
        if found_srcTensor.format == 'NCHW' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_FRACTAL_Z:
            return self.format_nchw_to_fz(input_data)
        if found_srcTensor.format == 'NCDHW' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NDC1HWC0:
            return self.format_ncdhw_to_ndc1hwc0(input_data)
        if found_srcTensor.format == 'NCDHW' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FRACTAL_Z_3D:
            return self.format_ncdhw_to_fz3d(input_data)
        if found_srcTensor.format == 'NC1HWC0' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NHWC:
            return self.format_nc1hwc0_to_nhwc(input_data)
        if found_srcTensor.format == 'NC1HWC0' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NCHW:
            return self.format_nc1hwc0_to_nchw(input_data)
        if found_srcTensor.format == 'HWCN' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_FRACTAL_Z:
            return self.format_hwcn_to_fz(input_data)
        if found_srcTensor.format == 'FRACTAL_Z' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NCHW:
            return self.format_fz_to_nchw(input_data)
        if found_srcTensor.format == 'FRACTAL_Z' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_HWCN:
            return self.format_fz_to_hwcn(input_data)
        if found_srcTensor.format == 'FRACTAL_Z_3D' and input_tmp["dstFormat"][0].value == AclFormat.ACL_FORMAT_NCDHW:
            return self.format_fz3d_to_ncdhw(input_data)

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

    def get_config_by_name(self, configs, target_name: str):
        for item in configs:
            if isinstance(item, list):
                # 如果是列表，检查第一个元素的name
                if item and hasattr(item[0], 'name') and item[0].name == target_name:
                    return item  # 返回整个列表
            elif hasattr(item, 'name') and item.name == target_name:
                return item  # 返回单个对象

        return None

    def format_nz_to_nd(self, input_data: InputDataset):

        # ===== 输入参数 =====
        src = input_data.kwargs["srcTensor"]                    # 逻辑上 ND
        want_dst_format = input_data.kwargs.get("dstFormat", 2) # 目标 ND=2
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        if want_dst_format != 2:
            print(f"[WARN] dstFormat={want_dst_format} != ND(2). 强制为 ND(2).")
        dst_format = 2

        # ===== 归一化到 (H, N, C) 并计算对齐 =====
        in_shape = tuple(int(d) for d in src.shape)
        in_len = len(in_shape)
        if in_len == 1:
            axis_h, axis_n, axis_c = 1, 1, in_shape[0]
        elif in_len == 2:
            axis_h, axis_n, axis_c = 1, in_shape[0], in_shape[1]
        else:
            axis_h = 1 if in_len <= 2 else int(torch.tensor(in_shape[:-2]).prod().item())
            axis_n = in_shape[-2]
            axis_c = in_shape[-1]

        C0 = self._c0_of_torch_dtype(src.dtype)
        NI = 16
        C1 = self._ceil_div(axis_c, C0)
        NO = self._ceil_div(axis_n, NI)
        n_pad = self._pad_len(axis_n, NI)
        c_pad = self._pad_len(axis_c, C0)

        # ===== 构造 NZ 的“真实存储”（5D: H, C1, NO, NI, C0），放到 NPU =====
        x = src.detach().to("cpu").contiguous().view(axis_h, axis_n, axis_c)
        if n_pad or c_pad:
            x = F.pad(x, (0, c_pad, 0, n_pad, 0, 0))
        nz5_cpu = x.view(axis_h, NO, NI, C1, C0).permute(0, 3, 1, 2, 4).contiguous()  # (H, C1, NO, NI, C0)
        src_storage = nz5_cpu.to("npu")  # 真实存储

        # ===== dtype 映射 =====
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ===== 逻辑视图与步长（view=原始 ND）=====
        view_shape = in_shape
        view_strides = self._contiguous_strides(view_shape)

        # ===== srcTensor: view=ND，storage=NZ(5D)，format=FRACTAL_NZ =====
        src_storage_shape = (axis_h, C1, NO, NI, C0)
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_FRACTAL_NZ,                                  # NZ
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ===== 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 ND）=====
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p   = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz  = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),         # 目标 ND
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ===== dstTensor: view=ND，storage=calc 返回（一般也是 ND），format=actualFormat =====
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        # dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ===== 返回（shape 传逻辑 view 维度列表！）=====
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr,  dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_nhwc_to_nc1hwc0(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]  # 逻辑视图：NCHW
        want_dst_format = input_data.kwargs.get("dstFormat", 3)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        N, H, W, C = map(int, src.shape)

        # 本场景固定期望 FZ=4
        if want_dst_format != 3:
            print(f"[WARN] dstFormat={want_dst_format} != FZ(4). 强制为 4.")
        dst_format = 3
        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 NCHW 视图）----
        view_shape = (N, H, W, C)
        view_strides = self._contiguous_strides(view_shape)
        src_storage = src.npu()
        # ---- srcTensor: view=NCDHW, storage=FZ3D(4D), format=FZ3D ----
        src_storage_shape = (N, H, W, C)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NHWC,  # ⚠️ NCHW
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCDHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=FZ, storage=Calc 返回（一般也是 FZ），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_nd_to_nz(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]
        dst_format = input_data.kwargs.get("dstFormat", 29)  # 默认 NDC1HWC0
        add_dtype = input_data.kwargs["additionalDtype"]

        # 1) 源张量：沿用你现有 backend 的封装（通常会把 torch.Tensor 封到 aclTensor*）
        srcTensor = self.backend.torch_tensor_to_acl(src, AclFormat.ACL_FORMAT_ND)

        # 2) 绑定 C 接口：aclnnNpuFormatCastCalculateSizeAndFormat
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,  # dstFormat
                ctypes.c_int,  # additionalDtype
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),  # int64_t **dstShape
                ctypes.POINTER(ctypes.c_uint64),  # uint64_t *dstShapeSize
                ctypes.POINTER(ctypes.c_int),  # int *actualFormat
            ],
            AclnnStatus
        )

        # 3) 申请出参并调用
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()  # int64_t*
        dst_shape_size = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            srcTensor.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_size),
            ctypes.byref(actual_format)
        )

        shape_len = int(dst_shape_size.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("aclnnNpuFormatCastCalculateSizeAndFormat returned empty dstShape")

        dst_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # 4) 目标 dtype：直接沿用 src.dtype；ACL dtype 用 TORCH_TO_ACLTYPE 做映射
        dst_torch_dtype = src.dtype
        try:
            dst_acl_dtype = TORCH_TO_ACLTYPE[str(dst_torch_dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {dst_torch_dtype}")

        # 5) 在 NPU 上按目标 shape 直接创建 torch 张量（连续）
        #    注意：这样 storage/strides 都是按 dst_shape 连续的，更符合 nnopbase.create_acl_tensor 的用法
        dst_storage = torch.empty(dst_shape, dtype=dst_torch_dtype, device="npu")

        # 6) 用 nnopbase 的封装创建 aclTensor（避免你手动算 strides/storageShape/ptr）
        dstTensorStruct: AclTensorStruct = nnopbase.create_acl_tensor(
            dst_storage, AclFormat(actual_format.value)
        )

        # 7) 组织返回参数
        numel = int(dst_storage.numel())
        size_u64 = Uint64(numel)

        input_args = [srcTensor, dstTensorStruct]  # 让后续 create_x_list 自行抽取 .tensor
        output_packages = [AclTensorStruct(
            dstTensorStruct.tensor, dstTensorStruct.addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]

        return input_args, output_packages

    def format_ndhwc_to_ndc1hwc0(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]  # 逻辑视图：NDHWC
        want_dst_format = input_data.kwargs.get("dstFormat", 32)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        N, D, H, W, C = map(int, src.shape)

        # 本场景固定期望 FZ=4
        if want_dst_format != 32:
            print(f"[WARN] dstFormat={want_dst_format} != NDC1HWC0(32). 强制为 32.")
        dst_format = 32
        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 NDHWC 视图）----
        view_shape = (N, D, H, W, C)
        view_strides = self._contiguous_strides(view_shape)
        src_storage = src.npu()
        # ---- srcTensor: view=NDHWC, storage=NDC1HWC0(6D), format=NDC1HWC0 ----
        src_storage_shape = (N, D, H, W, C)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NDHWC,  # ⚠️ NDHWC
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCDHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=FZ, storage=Calc 返回（一般也是 FZ），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_ndc1hwc0_to_ndhwc(self, input_data: InputDataset):
        # ===== 输入参数 =====
        src = input_data.kwargs["srcTensor"]  # torch.Tensor (N,C,D,H,W) 作为“逻辑视图”
        want_dst_format = input_data.kwargs.get("dstFormat", 27)  # 方向：NDC1HWC0 -> NCDHW，目标必须是 30
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        # 强制覆盖成 NCDHW，避免上游误传 32 造成方向不符
        if want_dst_format != 27:
            print(f"[WARN] dstFormat={want_dst_format} != NCDHW(30). 本测试场景强制改为 30(NCDHW).")
        dst_format = 27

        # ===== 计算 C0/C1、准备 NDC1HWC0 的底层存储（和 C++ 示例一致）=====
        N, D, H, W, C = map(int, src.shape)  # 逻辑视图 NCDHW
        C0 = self._c0_of_torch_dtype(src.dtype)
        C1 = self._ceil_div(C, C0)
        c_pad = C1 * C0 - C

        # 做一份“真实 NDC1HWC0”的内存，用作 srcTensor 的 storage（放到 NPU 上）
        x_cpu = src.detach().to("cpu").contiguous()
        if c_pad > 0:
            # pad 格式 (Wl,Wr, Hl,Hr, Dl,Dr, Cl,Cr, Nl,Nr)；只在 C 右侧补零
            x_cpu = F.pad(x_cpu, (c_pad, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        # (N,C,D,H,W) -> (N,C1,C0,D,H,W) -> (N,D,C1,H,W,C0)
        src_ndc1hwc0_cpu = x_cpu.view(N, D, H, W, C1, C0).permute(0, 1, 4, 2, 3, 5).contiguous()
        src_storage = src_ndc1hwc0_cpu.to("npu")

        # ===== dtype 映射 =====
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ===== 视图/存储形状、步长 =====
        view_shape = (N, D, H, W, C)  # 两边的“逻辑视图”统一用 NCDHW
        view_strides = self._contiguous_strides(view_shape)
        src_storage_shape = (N, D, C1, H, W, C0)  # 真实 NDC1HWC0

        # ====== 创建 srcTensor ：view=NCDHW, storage=NDC1HWC0, format=NDC1HWC0 ======
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NDC1HWC0,  # ⚠️ 与 C++ 示例一致：format 给真实存储的格式
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ====== 调一次 CalculateSizeAndFormat 拿到 dst 的 storage shape 与 actualFormat ======
        #     绑定签名（只需一次；若你全局绑定过可移除）
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),  # int64_t **dstShape
                ctypes.POINTER(ctypes.c_uint64),  # uint64_t *dstShapeSize
                ctypes.POINTER(ctypes.c_int),  # int *actualFormat
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        # 手动检查
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ====== 为 dst 分配 NPU 存储（按 storageShape）并创建 dstTensor ======
        #     注意：viewShape 依然是 NCDHW；format 用 actualFormat（通常是 NCDHW=30）
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())

        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ====== 组织返回（⚠️ shape 一定是逻辑视图的维度列表，不是 [numel]）=====
        input_args = [src_struct, dst_struct]  # 交给 create_x_list 抽 .tensor
        output_packages = [AclTensorStruct(
            dst_tensor,
            dst_addr,
            dst_storage,
            dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_ndc1hwc0_to_ncdhw(self, input_data: InputDataset):
        # ===== 输入参数 =====
        src = input_data.kwargs["srcTensor"]  # torch.Tensor (N,C,D,H,W) 作为“逻辑视图”
        want_dst_format = input_data.kwargs.get("dstFormat", 30)  # 方向：NDC1HWC0 -> NCDHW，目标必须是 30
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        # 强制覆盖成 NCDHW，避免上游误传 32 造成方向不符
        if want_dst_format != 30:
            print(f"[WARN] dstFormat={want_dst_format} != NCDHW(30). 本测试场景强制改为 30(NCDHW).")
        dst_format = 30

        # ===== 计算 C0/C1、准备 NDC1HWC0 的底层存储（和 C++ 示例一致）=====
        N, C, D, H, W = map(int, src.shape)  # 逻辑视图 NCDHW
        C0 = self._c0_of_torch_dtype(src.dtype)
        C1 = self._ceil_div(C, C0)
        c_pad = C1 * C0 - C

        # 做一份“真实 NDC1HWC0”的内存，用作 srcTensor 的 storage（放到 NPU 上）
        x_cpu = src.detach().to("cpu").contiguous()
        if c_pad > 0:
            # pad 格式 (Wl,Wr, Hl,Hr, Dl,Dr, Cl,Cr, Nl,Nr)；只在 C 右侧补零
            x_cpu = F.pad(x_cpu, (0, 0, 0, 0, 0, 0, 0, c_pad, 0, 0))
        # (N,C,D,H,W) -> (N,C1,C0,D,H,W) -> (N,D,C1,H,W,C0)
        src_ndc1hwc0_cpu = x_cpu.view(N, C1, C0, D, H, W).permute(0, 3, 1, 4, 5, 2).contiguous()
        src_storage = src_ndc1hwc0_cpu.to("npu")

        # ===== dtype 映射 =====
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ===== 视图/存储形状、步长 =====
        view_shape = (N, C, D, H, W)  # 两边的“逻辑视图”统一用 NCDHW
        view_strides = self._contiguous_strides(view_shape)
        src_storage_shape = (N, D, C1, H, W, C0)  # 真实 NDC1HWC0

        # ====== 创建 srcTensor ：view=NCDHW, storage=NDC1HWC0, format=NDC1HWC0 ======
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NDC1HWC0,  # ⚠️ 与 C++ 示例一致：format 给真实存储的格式
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ====== 调一次 CalculateSizeAndFormat 拿到 dst 的 storage shape 与 actualFormat ======
        #     绑定签名（只需一次；若你全局绑定过可移除）
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),  # int64_t **dstShape
                ctypes.POINTER(ctypes.c_uint64),  # uint64_t *dstShapeSize
                ctypes.POINTER(ctypes.c_int),  # int *actualFormat
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        # 手动检查
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ====== 为 dst 分配 NPU 存储（按 storageShape）并创建 dstTensor ======
        #     注意：viewShape 依然是 NCDHW；format 用 actualFormat（通常是 NCDHW=30）
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())

        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ====== 组织返回（⚠️ shape 一定是逻辑视图的维度列表，不是 [numel]）=====
        input_args = [src_struct, dst_struct]  # 交给 create_x_list 抽 .tensor
        output_packages = [AclTensorStruct(
            dst_tensor,
            dst_addr,
            dst_storage,
            dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_nchw_to_nc1hwc0(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]  # 逻辑视图：NCHW
        want_dst_format = input_data.kwargs.get("dstFormat", 3)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        N, C, H, W = map(int, src.shape)

        # 本场景固定期望 FZ=4
        if want_dst_format != 3:
            print(f"[WARN] dstFormat={want_dst_format} != FZ(4). 强制为 4.")
        dst_format = 3
        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 NCHW 视图）----
        view_shape = (N, C, H, W)
        view_strides = self._contiguous_strides(view_shape)
        src_storage = src.npu()
        # ---- srcTensor: view=NCDHW, storage=FZ3D(4D), format=FZ3D ----
        src_storage_shape = (N, C, H, W)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NCHW,  # ⚠️ NCHW
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCDHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=FZ, storage=Calc 返回（一般也是 FZ），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_nchw_to_fz(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]  # 逻辑视图：NCHW
        want_dst_format = input_data.kwargs.get("dstFormat", 4)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        N, C, H, W = map(int, src.shape)

        # 本场景固定期望 FZ=4
        if want_dst_format != 4:
            print(f"[WARN] dstFormat={want_dst_format} != FZ(4). 强制为 4.")
        dst_format = 4
        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 NCHW 视图）----
        view_shape = (N, C, H, W)
        view_strides = self._contiguous_strides(view_shape)
        src_storage = src.npu()
        # ---- srcTensor: view=NCDHW, storage=FZ3D(4D), format=FZ3D ----
        src_storage_shape = (N, C, H, W)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NCHW,  # ⚠️ NCHW
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCDHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=FZ, storage=Calc 返回（一般也是 FZ），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_ncdhw_to_ndc1hwc0(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]  # 逻辑视图：NCDHW
        want_dst_format = input_data.kwargs.get("dstFormat", 32)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        N, C, D, H, W = map(int, src.shape)

        # 本场景固定期望 FZ=4
        if want_dst_format != 32:
            print(f"[WARN] dstFormat={want_dst_format} != NDC1HWC0(32). 强制为 32.")
        dst_format = 32
        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 NCDHW 视图）----
        view_shape = (N, C, D, H, W)
        view_strides = self._contiguous_strides(view_shape)
        src_storage = src.npu()
        # ---- srcTensor: view=NCDHW, storage=FZ3D(4D), format=FZ3D ----
        src_storage_shape = (N, C, D, H, W)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NCDHW,  # ⚠️ NCDHW
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCDHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=FZ, storage=Calc 返回（一般也是 FZ），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_ncdhw_to_fz3d(self, input_data: InputDataset):

        src = input_data.kwargs["srcTensor"]  # 逻辑视图：NCDHW
        want_dst_format = input_data.kwargs.get("dstFormat", 33)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        N, C, D, H, W = map(int, src.shape)

        # 本场景固定期望 FZ=4
        if want_dst_format != 33:
            print(f"[WARN] dstFormat={want_dst_format} != FRACTAL_Z_3D(33). 强制为 33.")
        dst_format = 33
        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 NCDHW 视图）----
        view_shape = (N, C, D, H, W)
        view_strides = self._contiguous_strides(view_shape)
        src_storage = src.npu()
        # ---- srcTensor: view=NCDHW, storage=FZ3D(4D), format=FZ3D ----
        src_storage_shape = (N, C, D, H, W)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NCDHW,  # ⚠️ NCDHW
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCDHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=FZ, storage=Calc 返回（一般也是 FZ），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_nc1hwc0_to_nhwc(self, input_data: InputDataset):
        # ===== 输入参数 =====
        src = input_data.kwargs["srcTensor"]  # torch.Tensor (N,C,D,H,W) 作为“逻辑视图”
        want_dst_format = input_data.kwargs.get("dstFormat", 1)  # 方向：NDC1HWC0 -> NCDHW，目标必须是 30
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        # 强制覆盖成 NCDHW，避免上游误传 32 造成方向不符
        if want_dst_format != 1:
            print(f"[WARN] dstFormat={want_dst_format} != NCHW(0). 本测试场景强制改为 0(NCHW).")
        dst_format = 1

        # ===== 计算 C0/C1、准备 NDC1HWC0 的底层存储（和 C++ 示例一致）=====
        N, H, W, C = map(int, src.shape)  # 逻辑视图 NCDHW
        C0 = self._c0_of_torch_dtype(src.dtype)
        C1 = self._ceil_div(C, C0)
        c_pad = C1 * C0 - C

        # 做一份“真实 NDC1HWC0”的内存，用作 srcTensor 的 storage（放到 NPU 上）
        x_cpu = src.detach().to("cpu").contiguous()
        if c_pad > 0:
            # pad 格式 (Wl,Wr, Hl,Hr, Dl,Dr, Cl,Cr, Nl,Nr)；只在 C 右侧补零
            x_cpu = F.pad(x_cpu, (c_pad, 0, 0, 0, 0, 0, 0, 0))
        # (N,C,D,H,W) -> (N,C1,C0,D,H,W) -> (N,D,C1,H,W,C0)
        src_nc1hwc0_cpu = x_cpu.view(N, H, W, C1, C0).permute(0, 3, 1, 2, 4).contiguous()
        src_storage = src_nc1hwc0_cpu.to("npu")

        # ===== dtype 映射 =====
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ===== 视图/存储形状、步长 =====
        view_shape = (N, H, W, C)  # 两边的“逻辑视图”统一用 NCDHW
        view_strides = self._contiguous_strides(view_shape)
        src_storage_shape = (N, C1, H, W, C0)  # 真实 NDC1HWC0

        # ====== 创建 srcTensor ：view=NCDHW, storage=NDC1HWC0, format=NDC1HWC0 ======
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NC1HWC0,  # ⚠️ 与 C++ 示例一致：format 给真实存储的格式
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ====== 调一次 CalculateSizeAndFormat 拿到 dst 的 storage shape 与 actualFormat ======
        #     绑定签名（只需一次；若你全局绑定过可移除）
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),  # int64_t **dstShape
                ctypes.POINTER(ctypes.c_uint64),  # uint64_t *dstShapeSize
                ctypes.POINTER(ctypes.c_int),  # int *actualFormat
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        # 手动检查
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ====== 为 dst 分配 NPU 存储（按 storageShape）并创建 dstTensor ======
        #     注意：viewShape 依然是 NCDHW；format 用 actualFormat（通常是 NCDHW=30）
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())

        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ====== 组织返回（⚠️ shape 一定是逻辑视图的维度列表，不是 [numel]）=====
        input_args = [src_struct, dst_struct]  # 交给 create_x_list 抽 .tensor
        output_packages = [AclTensorStruct(
            dst_tensor,
            dst_addr,
            dst_storage,
            dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_nc1hwc0_to_nchw(self, input_data: InputDataset):
        # ===== 输入参数 =====
        src = input_data.kwargs["srcTensor"]  # torch.Tensor (N,C,D,H,W) 作为“逻辑视图”
        want_dst_format = input_data.kwargs.get("dstFormat", 0)  # 方向：NDC1HWC0 -> NCDHW，目标必须是 30
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        # 强制覆盖成 NCDHW，避免上游误传 32 造成方向不符
        if want_dst_format != 0:
            print(f"[WARN] dstFormat={want_dst_format} != NCHW(0). 本测试场景强制改为 0(NCHW).")
        dst_format = 0

        # ===== 计算 C0/C1、准备 NDC1HWC0 的底层存储（和 C++ 示例一致）=====
        N, C, H, W = map(int, src.shape)  # 逻辑视图 NCDHW
        C0 = self._c0_of_torch_dtype(src.dtype)
        C1 = self._ceil_div(C, C0)
        c_pad = C1 * C0 - C

        # 做一份“真实 NDC1HWC0”的内存，用作 srcTensor 的 storage（放到 NPU 上）
        x_cpu = src.detach().to("cpu").contiguous()
        if c_pad > 0:
            # pad 格式 (Wl,Wr, Hl,Hr, Dl,Dr, Cl,Cr, Nl,Nr)；只在 C 右侧补零
            x_cpu = F.pad(x_cpu, (0, 0, 0, 0, 0, c_pad, 0, 0))
        # (N,C,D,H,W) -> (N,C1,C0,D,H,W) -> (N,D,C1,H,W,C0)
        src_nc1hwc0_cpu = x_cpu.view(N, C1, C0, H, W).permute(0, 1, 3, 4, 2).contiguous()
        src_storage = src_nc1hwc0_cpu.to("npu")

        # ===== dtype 映射 =====
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ===== 视图/存储形状、步长 =====
        view_shape = (N, C, H, W)  # 两边的“逻辑视图”统一用 NCDHW
        view_strides = self._contiguous_strides(view_shape)
        src_storage_shape = (N, C1, H, W, C0)  # 真实 NDC1HWC0

        # ====== 创建 srcTensor ：view=NCDHW, storage=NDC1HWC0, format=NDC1HWC0 ======
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_NC1HWC0,  # ⚠️ 与 C++ 示例一致：format 给真实存储的格式
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ====== 调一次 CalculateSizeAndFormat 拿到 dst 的 storage shape 与 actualFormat ======
        #     绑定签名（只需一次；若你全局绑定过可移除）
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),  # int64_t **dstShape
                ctypes.POINTER(ctypes.c_uint64),  # uint64_t *dstShapeSize
                ctypes.POINTER(ctypes.c_int),  # int *actualFormat
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        # 手动检查
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ====== 为 dst 分配 NPU 存储（按 storageShape）并创建 dstTensor ======
        #     注意：viewShape 依然是 NCDHW；format 用 actualFormat（通常是 NCDHW=30）
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())

        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ====== 组织返回（⚠️ shape 一定是逻辑视图的维度列表，不是 [numel]）=====
        input_args = [src_struct, dst_struct]  # 交给 create_x_list 抽 .tensor
        output_packages = [AclTensorStruct(
            dst_tensor,
            dst_addr,
            dst_storage,
            dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_hwcn_to_fz(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]  # 逻辑视图：HWCN
        want_dst_format = input_data.kwargs.get("dstFormat", 4)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)
        H, W, C, N = map(int, src.shape)

        # 本场景固定期望 FZ=4
        if want_dst_format != 4:
            print(f"[WARN] dstFormat={want_dst_format} != FZ(4). 强制为 4.")
        dst_format = 4
        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 HWCN 视图）----
        view_shape = (H, W, C, N)
        view_strides = self._contiguous_strides(view_shape)
        src_storage = src.npu()
        # ---- srcTensor: view=HWCN, storage=NWCH, format=NWCH ----
        src_storage_shape = (H, W, C, N)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_HWCN,  # ⚠️ HWCN
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=FZ, storage=Calc 返回（一般也是 FZ），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_fz_to_nchw(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]  # 逻辑视图：NCHW
        want_dst_format = input_data.kwargs.get("dstFormat", 0)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        N, C, H, W = map(int, src.shape)
        C0 = self._c0_of_torch_dtype(src.dtype)
        if (src.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        NI = 16
        C1 = self._ceil_div(C, C0)
        NO = self._ceil_div(N, NI)
        n_pad = self._pad_len(N, NI)
        c_pad = self._pad_len(C, C0)

        x = src.detach().to("cpu").contiguous()
        if n_pad or c_pad:
            x = F.pad(x, (0, 0, 0, 0, 0, c_pad, 0, n_pad))  # (Wl,Wr, Hl,Hr, Cl,Cr, Nl,Nr)

        # (N, C, H, W) -> (NO, NI, C1, C0, H, W) -> (C1, H, W, NO, NI, C0)  [FZ3D 6D]
        x = x.view(NO, NI, C1, C0, H, W)
        z_6d = x.permute(2, 4, 5, 0, 1, 3).contiguous()

        src_storage = z_6d.reshape(C1 * H * W, NO, NI, C0).to("npu")

        # 本场景固定期望 FZ=4
        if want_dst_format != 0:
            print(f"[WARN] dstFormat={want_dst_format} != FZ(4). 强制为 4.")
        dst_format = 0
        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 NCHW 视图）----
        view_shape = (N, C, H, W)
        view_strides = self._contiguous_strides(view_shape)

        # ---- srcTensor: view=NCDHW, storage=FZ3D(4D), format=FZ3D ----
        src_storage_shape = (C1 * H * W, NO, NI, C0)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_FRACTAL_Z,  # ⚠️ NCHW
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCDHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=FZ, storage=Calc 返回（一般也是 FZ），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_fz_to_hwcn(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]  # 逻辑视图：NCHW
        want_dst_format = input_data.kwargs.get("dstFormat", 16)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        H, W, C, N = map(int, src.shape)
        C0 = self._c0_of_torch_dtype(src.dtype)
        if (src.dtype in (torch.int32, torch.float32, torch.uint32) and C0 == 16):
            self.additionalDtypeChoice = random.choice([1, 27])
        NI = 16
        C1 = self._ceil_div(C, C0)
        NO = self._ceil_div(N, NI)
        n_pad = self._pad_len(N, NI)
        c_pad = self._pad_len(C, C0)

        x = src.detach().to("cpu").contiguous()
        if n_pad or c_pad:
            x = F.pad(x, (0, n_pad, 0, c_pad, 0, 0, 0, 0))  # (Wl,Wr, Hl,Hr, Cl,Cr, Nl,Nr)

        # (N, C, H, W) -> (NO, NI, C1, C0, H, W) -> (C1, H, W, NO, NI, C0)  [FZ3D 6D]
        x = x.view(H, W, C1, C0, NO, NI)
        z_6d = x.permute(2, 0, 1, 4, 5, 3).contiguous()

        src_storage = z_6d.reshape(C1 * H * W, NO, NI, C0).to("npu")

        # 本场景固定期望 FZ=4
        if want_dst_format != 16:
            print(f"[WARN] dstFormat={want_dst_format} != FZ(4). 强制为 4.")
        dst_format = 16
        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 NCHW 视图）----
        view_shape = (H, W, C, N)
        view_strides = self._contiguous_strides(view_shape)

        # ---- srcTensor: view=NCDHW, storage=FZ3D(4D), format=FZ3D ----
        src_storage_shape = (C1 * H * W, NO, NI, C0)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FORMAT_FRACTAL_Z,  # ⚠️ NCHW
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCDHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=FZ, storage=Calc 返回（一般也是 FZ），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages

    def format_fz3d_to_ncdhw(self, input_data: InputDataset):
        src = input_data.kwargs["srcTensor"]  # 逻辑视图：NCDHW
        want_dst_format = input_data.kwargs.get("dstFormat", 30)
        add_dtype = input_data.kwargs.get("additionalDtype", -1)

        # 本场景固定期望 NCDHW=30
        if want_dst_format != 30:
            print(f"[WARN] dstFormat={want_dst_format} != NCDHW(30). 强制为 30.")
        dst_format = 30

        # ---- 计算 NO/NI/C1/C0，打包出 FZ3D 存储（4D：DC1HW × NO × NI × C0）----
        N, C, D, H, W = map(int, src.shape)
        C0 = self._c0_of_torch_dtype(src.dtype)
        NI = 16
        C1 = self._ceil_div(C, C0)
        NO = self._ceil_div(N, NI)
        n_pad = self._pad_len(N, NI)
        c_pad = self._pad_len(C, C0)

        x_cpu = src.detach().to("cpu").contiguous()
        if n_pad or c_pad:
            x_cpu = F.pad(x_cpu, (0, 0, 0, 0, 0, 0, 0, c_pad, 0, n_pad))

        # (N,C,D,H,W)->(NO,NI,C1,C0,D,H,W)->(D,C1,H,W,NO,NI,C0) -> 4D FZ3D storage
        x_cpu = x_cpu.view(NO, NI, C1, C0, D, H, W)
        z3d_7d = x_cpu.permute(4, 2, 5, 6, 0, 1, 3).contiguous()
        src_storage = z3d_7d.reshape(D * C1 * H * W, NO, NI, C0).to("npu")  # 真实 FZ3D 存储

        # ---- dtype 映射 ----
        try:
            acl_dtype = TORCH_TO_ACLTYPE[str(src.dtype)]
        except KeyError:
            raise ValueError(f"Unsupported torch dtype for ACL: {src.dtype}")

        # ---- 逻辑视图与步长（两端都用 NCDHW 视图）----
        view_shape = (N, C, D, H, W)
        view_strides = self._contiguous_strides(view_shape)

        # ---- srcTensor: view=NCDHW, storage=FZ3D(4D), format=FZ3D ----
        src_storage_shape = (D * C1 * H * W, NO, NI, C0)  # golden 的 in_shape 定义
        src_addr = ctypes.c_void_p(src_storage.data_ptr())
        src_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat.ACL_FRACTAL_Z_3D,  # ⚠️ FZ3D
            (Int64 * len(src_storage_shape))(*src_storage_shape), len(src_storage_shape),
            src_addr
        )
        src_struct = AclTensorStruct(src_tensor, src_addr, list(view_shape), acl_dtype)

        # ---- 调 CalcSizeAndFormat：拿 dst 的 storageShape 与 actualFormat（通常 NCDHW）----
        acl_wrapper.aclnn.bind_function(
            "aclnnNpuFormatCastCalculateSizeAndFormat",
            [
                TensorPtr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
            ],
            AclnnStatus
        )
        dst_shape_p = ctypes.POINTER(ctypes.c_int64)()
        dst_shape_sz = ctypes.c_uint64(0)
        actual_format = ctypes.c_int(0)

        ret = acl_wrapper.aclnn.aclnnNpuFormatCastCalculateSizeAndFormat(
            src_struct.tensor,
            ctypes.c_int(dst_format),
            ctypes.c_int(add_dtype),
            ctypes.byref(dst_shape_p),
            ctypes.byref(dst_shape_sz),
            ctypes.byref(actual_format)
        )
        if ret.value != AclnnStatus.ACLNN_SUCCESS:
            raise RuntimeError(f"CalculateSizeAndFormat failed, ret={int(ret.value)}")

        shape_len = int(dst_shape_sz.value)
        if shape_len <= 0 or not bool(dst_shape_p):
            raise RuntimeError("CalculateSizeAndFormat returned empty dstShape")

        dst_storage_shape = tuple(int(dst_shape_p[i]) for i in range(shape_len))

        # ---- dstTensor: view=NCDHW, storage=Calc 返回（一般也是 NCDHW），format=actualFormat ----
        dst_storage = torch.empty(dst_storage_shape, dtype=src.dtype, device="npu")
        dst_addr = ctypes.c_void_p(dst_storage.data_ptr())
        dst_tensor = nnopbase.aclCreateTensor(
            (Int64 * len(view_shape))(*view_shape), len(view_shape),
            acl_dtype,
            (Int64 * len(view_strides))(*view_strides), 0,
            AclFormat(actual_format.value),
            (Int64 * len(dst_storage_shape))(*dst_storage_shape), len(dst_storage_shape),
            dst_addr
        )
        dst_struct = AclTensorStruct(dst_tensor, dst_addr, list(view_shape), acl_dtype)

        # ---- 返回（shape 传逻辑视图维度列表！）----
        input_args = [src_struct, dst_struct]
        output_packages = [AclTensorStruct(
            dst_tensor, dst_addr, dst_storage, dst_storage.element_size() * dst_storage.numel()
        )]
        return input_args, output_packages