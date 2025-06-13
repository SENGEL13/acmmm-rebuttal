from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


# @triton.jit
# def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
#     """
#     Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

#     Args:
#         x_ptr (triton.Pointer): Pointer to the input tensor.
#         y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
#         s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
#         BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

#     Returns:
#         None
#     """
#     pid = tl.program_id(axis=0)
#     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     x = tl.load(x_ptr + offs).to(tl.float32)
#     s = tl.max(tl.abs(x)) / 448.
#     y = x / s
#     y = y.to(y_ptr.dtype.element_ty)
#     tl.store(y_ptr + offs, y)
#     tl.store(s_ptr + pid, s)

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr[ and stores the result in ](file://acmmm/kernal.py#11#38)y_ptr` using a fixed scaling factor.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s (tl.tensor): The scaling factor value to use for quantization.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    # 使用传入的标量缩放因子
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)


def dynamic_act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    s = (torch.max(torch.abs(x)).clamp(1e-7) / 128).item()
    y = torch.empty_like(x, dtype=torch.int8)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s

def static_act_quant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=torch.int8)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s

fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

# @triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
# @triton.jit
# def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
#                     a_s_ptr, b_s_ptr,
#                     M, N: tl.constexpr, K: tl.constexpr,
#                     BLOCK_SIZE_M: tl.constexpr,
#                     BLOCK_SIZE_N: tl.constexpr,
#                     BLOCK_SIZE_K: tl.constexpr):
#     """
#     Performs a matrix multiplication operation on FP8 matrices with scaling factors.

#     Args:
#         a_ptr (tl.tensor): Pointer to the first input matrix A.
#         b_ptr (tl.tensor): Pointer to the second input matrix B.
#         c_ptr (tl.tensor): Pointer to the output matrix C.
#         a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
#         b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
#         M (int): Number of rows in matrix A and C.
#         N (tl.constexpr): Number of columns in matrix B and C.
#         K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
#         BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
#         BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
#         BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

#     Returns:
#         None
#     """
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
#     k = tl.cdiv(K, BLOCK_SIZE_K)
#     offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
#     b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
#     a_s_ptrs = a_s_ptr + offs_m * k
#     b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#     for i in range(k):
#         a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
#         b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
#         a_s = tl.load(a_s_ptrs)
#         b_s = tl.load(b_s_ptrs)
#         accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
#         a_ptrs += BLOCK_SIZE_K
#         b_ptrs += BLOCK_SIZE_K
#         a_s_ptrs += 1
#         b_s_ptrs += 1
#     c = accumulator.to(c_ptr.dtype.element_ty)
#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
#     mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
#     tl.store(c_ptrs, c, mask=mask)

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def int8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s, b_s,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b) * a_s * b_s
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp32_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def int8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous()
    # assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    int8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c

def fp32_gemm(a: torch.Tensor, b: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    # assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp32_gemm_kernel[grid](a, b, c, M, N, K)
    return c

if __name__ == '__main__':
    a = torch.randn(1024, 1024).cuda()
    b = torch.randn(1024, 1024).cuda()
    print(a @ b)
    a_s = a.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / 16
    a_s = 1.
    b_s = 1.
    a_q, a_s = static_act_quant(a, a_s)
    b_q, b_s = static_act_quant(b, b_s)
    c = fp8_gemm(a_q, a_s, b_q, b_s)
    print(a_q, b_q, c)
