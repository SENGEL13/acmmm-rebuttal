from typing import Tuple
import time
import numpy as np
import torch
import triton
import triton.language as tl
from triton import Config
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import gc

# 保留您原有的函数实现
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)

def static_act_quant(x: torch.Tensor, s: float, block_size: int = 128) -> Tuple[torch.Tensor, float]:
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

# 新增的性能测试函数
def measure_time(func, *args, **kwargs):
    """测量函数执行时间"""
    torch.cuda.synchronize()
    start = time.time()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    return result, (end - start) * 1000  # 返回毫秒

def measure_memory_usage(func, *args, **kwargs):
    """测量函数执行的内存使用情况"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    
    start_mem = torch.cuda.max_memory_allocated()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    end_mem = torch.cuda.max_memory_allocated()
    
    memory_used = (end_mem - start_mem) / (1024 ** 2)  # MB
    return result, memory_used

def run_pytorch_matmul(a, b):
    """运行PyTorch原生矩阵乘法"""
    return fp32_gemm(a, b)

def run_quantized_matmul(a, b):
    """运行量化后的矩阵乘法"""
    a_s = 1.0
    b_s = 1.0
    a_q, a_s = static_act_quant(a, a_s)
    b_q, b_s = static_act_quant(b, b_s)
    return int8_gemm(a_q, a_s, b_q, b_s)

def profile_operations(sizes):
    """对不同大小的矩阵进行性能分析"""
    pytorch_times = []
    triton_times = []
    pytorch_memory = []
    triton_memory = []
    
    for size in sizes:
        print(f"Testing matrix size: {size}x{size}")
        
        # 创建测试矩阵
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # 预热
        _ = run_pytorch_matmul(a, b)
        _ = run_quantized_matmul(a, b)
        torch.cuda.synchronize()
        
        # 测量PyTorch矩阵乘法时间
        _, pt_time = measure_time(run_pytorch_matmul, a, b)
        pytorch_times.append(pt_time)
        
        # 测量PyTorch矩阵乘法内存
        _, pt_mem = measure_memory_usage(run_pytorch_matmul, a, b)
        pytorch_memory.append(pt_mem)
        
        # 测量Triton量化矩阵乘法时间
        _, tr_time = measure_time(run_quantized_matmul, a, b)
        triton_times.append(tr_time)
        
        # 测量Triton量化矩阵乘法内存
        _, tr_mem = measure_memory_usage(run_quantized_matmul, a, b)
        triton_memory.append(tr_mem)
        
        # 清理内存
        del a, b
        torch.cuda.empty_cache()
        gc.collect()
        
    return pytorch_times, triton_times, pytorch_memory, triton_memory

def verify_correctness(size=1024):
    """验证量化矩阵乘法的正确性"""
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    # 计算参考结果
    ref = a @ b
    
    # 计算量化结果
    a_s = 1.0
    b_s = 1.0
    a_q, a_s = static_act_quant(a, a_s)
    b_q, b_s = static_act_quant(b, b_s)
    res = int8_gemm(a_q, a_s, b_q, b_s)
    
    # 计算相对误差
    abs_diff = torch.abs(ref - res)
    rel_error = torch.mean(abs_diff / (torch.abs(ref) + 1e-7))
    max_error = torch.max(abs_diff)
    
    print(f"平均相对误差: {rel_error.item():.6f}")
    print(f"最大绝对误差: {max_error.item():.6f}")
    
    return rel_error.item(), max_error.item()

def plot_results(sizes, pytorch_times, triton_times, pytorch_memory, triton_memory):
    """绘制性能比较图表"""
    plt.figure(figsize=(15, 10))
    
    # 时间对比图
    plt.subplot(2, 1, 1)
    plt.plot(sizes, pytorch_times, 'o-', label='PyTorch')
    plt.plot(sizes, triton_times, 's-', label='Triton Quantized')
    plt.title('Execution Time Comparison')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    
    # 内存对比图
    plt.subplot(2, 1, 2)
    plt.plot(sizes, pytorch_memory, 'o-', label='PyTorch')
    plt.plot(sizes, triton_memory, 's-', label='Triton Quantized')
    plt.title('Memory Usage Comparison')
    plt.xlabel('Matrix Size')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('matmul_performance_comparison.png')
    plt.show()

def main():
    # 验证结果正确性
    print("验证计算结果正确性...")
    verify_correctness()
    
    # 测试不同大小的矩阵
    sizes = [512, 1024, 2048, 4096, 8192]
    print("\n开始性能测试...")
    pytorch_times, triton_times, pytorch_memory, triton_memory = profile_operations(sizes)
    
    # 输出结果
    print("\n性能测试结果:")
    print("矩阵大小\tPyTorch时间(ms)\tTriton时间(ms)\t加速比\tPyTorch内存(MB)\tTriton内存(MB)\t内存节省")
    for i, size in enumerate(sizes):
        speedup = pytorch_times[i] / triton_times[i]
        mem_saving = pytorch_memory[i] / triton_memory[i] if triton_memory[i] > 0 else float('inf')
        print(f"{size}x{size}\t{pytorch_times[i]:.2f}\t\t{triton_times[i]:.2f}\t\t{speedup:.2f}x\t{pytorch_memory[i]:.2f}\t\t{triton_memory[i]:.2f}\t\t{mem_saving:.2f}x")
    
    # 绘制结果图表
    plot_results(sizes, pytorch_times, triton_times, pytorch_memory, triton_memory)

if __name__ == '__main__':
    main()
