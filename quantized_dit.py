import sys
import os
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上上层目录的路径
package_dir1 = os.path.abspath(os.path.join(current_dir, '../../'))
package_dir2 = os.path.abspath(os.path.join(current_dir, '../../../'))

# 添加上上层目录到 sys.path
sys.path.append(package_dir1)
sys.path.append(package_dir2)

import torch
import torch.nn as nn
import quarot
from timm.layers.helpers import to_2tuple
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

    def forward(self, x, calib=False):
        B, N, C = x.shape  # [b=100, n=256, d=1152]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        if not calib and self.use_act_quant:
            attn = self.act_quantizer_q(q) @ self.act_quantizer_k(k).transpose(-2, -1)
        else:
            attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if not calib and self.use_act_quant:
            x = self.act_quantizer_w(attn) @ self.act_quantizer_v(v)
        else:
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
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

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
        x = self.quantizer(x)
        x = self.fc1_hadamard(x)
        return super().forward(x)

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
            Quantizer(),
            nn.SiLU(),
            Linear4bit.from_float(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
if __name__ == '__main__':
    model = FlattenDiTBlock(1152, 16)
    x = torch.randn(1, 256, 1152)
    c = torch.randn(1, 256, 1152)
    y = model(x, c)
