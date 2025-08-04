import triton
import triton.language as tl
import torch
from lucyrnn_conf import LucyRNNConfig
import torch.nn as nn

class LinearSafe(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.zeros_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Works with x of shape (B*T, D) or (B, T, D)
        x_flat = x.view(-1, x.shape[-1]).contiguous()
        out = torch.matmul(x_flat, self.weight.T)
        if self.bias is not None:
            out += self.bias
        return out.view(*x.shape[:-1], self.weight.shape[0])

class LucyRNNCellTriton(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = LinearSafe(input_dim, 6 * hidden_dim)

    def forward(self, x, h0, s0):
        B, T, _ = x.shape
        D = self.hidden_dim
        device = x.device

        # Precompute gates: shape [B, T, 6 * D]
        gates = self.linear(x)
        gates = gates.view(B, T, 6, D).contiguous()

        out = torch.empty(B, T, D, device=device, dtype=x.dtype)
        s_out = torch.empty(B, D, device=device, dtype=x.dtype)

        rnn_forward_unfused_rmsnorm[(B, D)](
            gates_ptr=gates,
            h0_ptr=h0,
            s0_ptr=s0,
            out_ptr=out,
            s_out_ptr=s_out,
            B=B, T=T, D=D,
            stride_g_bt=gates.stride(0),
            stride_g_td=gates.stride(1),
            stride_g_cd=gates.stride(2),
            stride_o_bt=out.stride(0),
            stride_o_bd=out.stride(1),
        )

        return out, s_out

class LucyRNNtriton(nn.Module):
    def __init__(self, config: LucyRNNConfig):
        super().__init__()
        assert config.fused_ops
        assert not config.layer_norm
        assert config.stack_order == 1
        assert config.decay_mode == 'learned'

        self.config = config
        self.layers = nn.ModuleList()

        for i in range(config.num_layers):
            input_dim = config.input_dim if i == 0 else config.hidden_dim
            self.layers.append(LucyRNNCellTriton(input_dim, config.hidden_dim))

        self.output_proj = LinearSafe(config.hidden_dim, config.vocab_size)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, hidden_states=None, masks=None):
        B, T, _ = x.shape
        dtype = x.dtype
        device = x.device

        if hidden_states is None:
            h = [torch.zeros(B, self.config.hidden_dim, device=device, dtype=dtype)
                 for _ in range(self.config.num_layers)]
            s = [torch.zeros(B, self.config.hidden_dim, device=device, dtype=dtype)
                 for _ in range(self.config.num_layers)]
        else:
            h, s = hidden_states

        for l, layer in enumerate(self.layers):
            x, s[l] = layer(x, h[l], s[l])
            h[l] = x[:, -1, :]

        # F.linear() expects proper contiguous layout for batched matmul - triton may not produce that
        x = x.contiguous()

        logits = self.output_proj(x)

        if self.config.return_last_states:
            return logits, (h, s)
        else:
            return logits

@triton.jit
def fused_decay_scan(
    kv_ptr, decay_ptr, output_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    stride_b: tl.constexpr, stride_t: tl.constexpr, stride_d: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1)

    offset_kv = b * stride_b + d
    offset_decay = offset_kv
    offset_out = offset_kv

    s = tl.zeros((), dtype=tl.float32)

    for t in range(T):
        kv = tl.load(kv_ptr + offset_kv + t * stride_t)
        decay = tl.load(decay_ptr + offset_decay + t * stride_t)
        s = decay * s + kv
        tl.store(output_ptr + offset_out + t * stride_t, s)

@triton.jit
def rnn_forward_unfused_rmsnorm(
    gates_ptr,  # [B, T, 6, D] flattened
    h0_ptr,     # [B, D]
    s0_ptr,     # [B, D]
    out_ptr,    # [B, T, D]
    s_out_ptr,  # [B, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    stride_g_bt: tl.constexpr,
    stride_g_td: tl.constexpr,
    stride_g_cd: tl.constexpr,
    stride_o_bt: tl.constexpr,
    stride_o_bd: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1)

    if b >= B or d >= D:
        return

    h = tl.load(h0_ptr + b * D + d)
    s = tl.load(s0_ptr + b * D + d)

    for t in range(T):
        # Load gates
        r = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 0 * stride_g_cd + d)
        z = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 1 * stride_g_cd + d)
        k = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 2 * stride_g_cd + d)
        v = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 3 * stride_g_cd + d)
        h_pre = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 4 * stride_g_cd + d)
        decay = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 5 * stride_g_cd + d)

        # ---------- RMSNorm over gates at this (b, t, d) ----------
        # rms = sqrt(mean([r, z, k, v, h_pre, decay]^2))
        sum_squares = r * r + z * z + k * k + v * v + h_pre * h_pre + decay * decay
        rms = tl.sqrt(sum_squares / 6 + 1e-6)

        # Normalize
        r /= rms
        z /= rms
        k /= rms
        v /= rms
        h_pre /= rms
        decay /= rms

        # Apply nonlinearities
        r = tl.sigmoid(r)
        z = tl.sigmoid(z)
        decay = tl.sigmoid(decay)

        kv = k * v
        s = decay * s + kv
        c = tl.sigmoid(2 * (h_pre + s)) * 2.0 - 1.0
        h = (1. - z) * c + z * h

        tl.store(out_ptr + b * stride_o_bt + t * stride_o_bd + d, h)

    tl.store(s_out_ptr + b * D + d, s)



