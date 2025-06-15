import torch
import torch.nn as nn
import quarot
# from timm.layers.helpers import to_2tuple
from functools import partial
from quarot.nn import Quantizer, Linear4bit, RMSNorm, OnlineHadamard
# from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fp16=True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale_qkv = None
        self.scale_proj = None
        self.time_step = 0
        self.fp16 = fp16
        if self.fp16:
            self.half()

    def forward(self, x, calib=False):
        B, N, C = x.shape  # [b=100, n=256, d=1152]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        # if not calib and self.use_act_quant:
        #     attn = self.act_quantizer_q(q) @ self.act_quantizer_k(k).transpose(-2, -1)
        # else:
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # if not calib and self.use_act_quant:
        #     x = self.act_quantizer_w(attn) @ self.act_quantizer_v(v)
        # else:
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)  # [b, n, d]

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FlattenAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 量化组件
        self.quantizer = Quantizer()
        self.qkv = Linear4bit.from_float(self.qkv)
        self.o_proj_hadamard = OnlineHadamard(self.num_heads)
        self.proj = nn.Sequential(
            Quantizer(),
            Linear4bit.from_float(self.proj)
        )

    def forward(self, x):
        B, N, C = x.shape
        
        # 量化输入
        x = self.quantizer(x)
        
        # QKV投影
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 注意力输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Hadamard变换和输出投影
        x = self.o_proj_hadamard(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.proj(x)
        
        return x
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            fp16=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        # bias = to_2tuple(bias)
        # drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)
        self.fp16 = fp16
        if self.fp16:
            self.half()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class FlattenMlp(Mlp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = Quantizer()
        self.fc1_hadamard = OnlineHadamard(self.in_features)
        self.fc1 = Linear4bit.from_float(self.fc1)
        self.fc2 = nn.Sequential(
            OnlineHadamard(self.hidden_features),
            Quantizer(),
            Linear4bit.from_float(self.fc2)
        )

    def forward(self, x):
        x = self.fc1_hadamard(x)
        x = self.quantizer(x)
        return super().forward(x)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True).half()
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FlattenDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = FlattenAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = FlattenMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            Quantizer(),
            Linear4bit.from_float(nn.Linear(hidden_size, 6 * hidden_size, bias=True).half())
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
# ... existing code ...

if __name__ == '__main__':
    import time
    import torch.cuda
    import gc
    from torch.profiler import profile, record_function, ProfilerActivity
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model_int4 = FlattenDiTBlock(1152, 16).to(device)
    model_fp16 = DiTBlock(1152, 16).to(device)
    
    # 创建测试数据
    batch_size = 4
    seq_len = 256
    hidden_size = 1152
    x = torch.randn(batch_size, seq_len, hidden_size).to(device)
    c = torch.randn(batch_size, hidden_size).to(device)
    
    # 预热
    print("预热模型...")
    for _ in range(10):
        with torch.no_grad():
            _ = model_int4(x, c)
            _ = model_fp16(x, c)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    def measure_memory_and_time(model, x, c, model_name, num_runs=100):
        """测量模型的内存使用和推理时间"""
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # 记录初始内存
        if device.type == 'cuda':
            memory_before = torch.cuda.memory_allocated()
        
        # 时间测试
        times = []
        model.eval()
        
        with torch.no_grad():
            for i in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                output = model(x, c)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        # 内存统计
        if device.type == 'cuda':
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = memory_after - memory_before
        else:
            memory_used = 0
            peak_memory = 0
        
        # 时间统计
        avg_time = sum(times) * 1000 / len(times)  # 转换为毫秒
        min_time = min(times) * 1000
        max_time = max(times) * 1000
        
        print(f"\n{model_name} 性能统计:")
        print(f"  平均推理时间: {avg_time:.2f} ms")
        print(f"  最小推理时间: {min_time:.2f} ms") 
        print(f"  最大推理时间: {max_time:.2f} ms")
        
        if device.type == 'cuda':
            print(f"  内存使用: {memory_used / 1024**2:.2f} MB")
            print(f"  峰值内存: {peak_memory / 1024**2:.2f} MB")
        
        return avg_time, memory_used, peak_memory
    
    def get_model_size(model):
        """计算模型参数大小"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        return total_size / 1024**2  # 转换为MB
    
    # 测试模型大小
    print("\n模型大小比较:")
    int4_size = get_model_size(model_int4)
    fp16_size = get_model_size(model_fp16)
    print(f"INT4 模型大小: {int4_size:.2f} MB")
    print(f"FP16 模型大小: {fp16_size:.2f} MB")
    print(f"压缩比: {fp16_size/int4_size:.2f}x")
    
    # 测试推理性能
    print("\n推理性能测试:")
    int4_time, int4_memory, int4_peak = measure_memory_and_time(
        model_int4, x, c, "INT4 模型", num_runs=100
    )
    
    fp16_time, fp16_memory, fp16_peak = measure_memory_and_time(
        model_fp16, x, c, "FP16 模型", num_runs=100
    )
    
    # 性能对比
    print(f"\n性能对比:")
    print(f"时延比较: INT4 vs FP16 = {int4_time:.2f} ms vs {fp16_time:.2f} ms")
    if fp16_time > 0:
        speedup = fp16_time / int4_time
        print(f"加速比: {speedup:.2f}x {'(INT4更快)' if speedup > 1 else '(FP16更快)'}")
    
    if device.type == 'cuda':
        print(f"内存使用比较: INT4 vs FP16 = {int4_memory/1024**2:.2f} MB vs {fp16_memory/1024**2:.2f} MB")
        if fp16_memory > 0:
            memory_ratio = fp16_memory / int4_memory if int4_memory > 0 else float('inf')
            print(f"内存节省: {memory_ratio:.2f}x")
    
    # 详细性能分析（可选）
    print(f"\n详细性能分析:")
    
    def detailed_profile(model, x, c, model_name):
        """使用PyTorch Profiler进行详细分析"""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == 'cuda' else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function(f"{model_name}_forward"):
                for _ in range(10):
                    model(x, c)
        
        print(f"\n{model_name} 详细分析:")
        print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == 'cuda' else "cpu_time_total", row_limit=10))
    
    # 运行详细分析
    detailed_profile(model_int4, x, c, "INT4")
    detailed_profile(model_fp16, x, c, "FP16")
