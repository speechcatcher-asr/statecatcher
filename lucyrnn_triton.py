import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from lucyrnn_conf import LucyRNNConfig
import torch.nn as nn

class LinearSafe(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
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
        self.linear = LinearSafe(input_dim, 7 * hidden_dim)  # 7 gates
        self._init_weights()

    def _init_weights(self):
        # Xavier init
        nn.init.xavier_uniform_(self.linear.weight)
        # Gate-aware bias inits
        if self.linear.bias is not None:
            D = self.hidden_dim
            with torch.no_grad():
                self.linear.bias[0*D:1*D].zero_()       # r gate
                self.linear.bias[1*D:2*D].fill_(1.0)    # z gate
                self.linear.bias[2*D:3*D].zero_()       # k
                self.linear.bias[3*D:4*D].zero_()       # v
                self.linear.bias[4*D:5*D].zero_()       # h_pre
                self.linear.bias[5*D:6*D].fill_(2.0)    # decay
                self.linear.bias[6*D:7*D].fill_(0.5)    # alpha gate

    def forward(self, x, h0, s0):
        B, T, _ = x.shape
        D = self.hidden_dim
        device = x.device

        # Compute gates: [B, T, 7 * D]
        gates = self.linear(x).view(B, T, 7, D).contiguous()

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
        assert not config.layer_norm  # your config disables per-gate LN
        assert config.stack_order == 1
        assert config.decay_mode == 'learned'

        self.config = config
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(config.num_layers):
            input_dim = config.input_dim if i == 0 else config.hidden_dim
            self.layers.append(LucyRNNCellTriton(input_dim, config.hidden_dim))

            # Add LayerNorm after each layer *except the last*
            if i < config.num_layers - 1:
                self.norms.append(nn.LayerNorm(config.hidden_dim))

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

            # Apply LayerNorm after each layer except the last
            if l < self.config.num_layers - 1:
                x = self.norms[l](x)

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
    gates_ptr,  # [B, T, 7, D] flattened
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
        r = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 0 * stride_g_cd + d)
        z = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 1 * stride_g_cd + d)
        k = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 2 * stride_g_cd + d)
        v = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 3 * stride_g_cd + d)
        h_pre = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 4 * stride_g_cd + d)
        decay = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 5 * stride_g_cd + d)
        alpha = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 6 * stride_g_cd + d)

        # Grouped RMSNorm
        rms_control = tl.sqrt((r*r + z*z) / 2 + 1e-6)
        rms_kv = tl.sqrt((k*k + v*v) / 2 + 1e-6)
        rms_decay = tl.sqrt(decay * decay + 1e-6)
        rms_alpha = tl.sqrt(alpha * alpha + 1e-6)
        rms_h = tl.sqrt(h_pre * h_pre + 1e-6)

        r /= rms_control
        z /= rms_control
        decay /= rms_decay
        k /= rms_kv
        v /= rms_kv
        h_pre /= rms_h
        alpha /= rms_alpha

        # Nonlinearities
        r = tl.sigmoid(r)
        z = tl.sigmoid(z)
        decay = tl.sigmoid(decay)
        alpha = tl.sigmoid(alpha)

        # Bounded kv update
        kv = (k * v) / (rms_kv * rms_kv + 1e-6)

        # Gated injection via alpha
        s = decay * s + alpha * kv
        c = tl.sigmoid(2 * (h_pre + s)) * 2.0 - 1.0
        h = (1. - z) * c + z * h

        tl.store(out_ptr + b * stride_o_bt + t * stride_o_bd + d, h)

    tl.store(s_out_ptr + b * D + d, s)

