import torch
import torch.nn as nn
from dataclasses import dataclass
from lucyrnn_triton import fused_decay_scan, fused_rnn_forward, rnn_forward_unfused
import torch.nn.functional as F

@dataclass
class LucyRNNConfig:
    input_dim: int
    hidden_dim: int
    num_layers: int
    vocab_size: int
    return_last_states: bool = True
    kernel_impl: str = "native"  # Options: 'native', 'triton'
    is_training: bool = True      # Determines parallel vs sequential implementation
    fused_ops: bool = False       # Use fused linear projections
    layer_norm: bool = True       # Enable/disable LayerNorm for benchmarking
    stack_order: int = 1          # Number of input frames to stack (e.g., 3 means [1,2,3], [4,5,6], ...)
    decay_mode: str = "learned"   # Options: 'learned', 'prefix_sum'
    lambda_decay: float = 0.001  # Used only for 'prefix_sum' decay

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

        rnn_forward_unfused[(B, D)](
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

class LucyRNNCellTritonFused(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_fused = nn.Linear(input_dim, 6 * hidden_dim)

    def forward(self, x, h0, s0):
        B, T, _ = x.shape
        D = self.hidden_dim
        device = x.device
        out = torch.empty(B, T, D, device=device, dtype=x.dtype)
        s_out = torch.empty(B, D, device=device, dtype=x.dtype)

        fused_rnn_forward_minimal[(B, D)](
            x_ptr=x,
            h0_ptr=h0,
            s0_ptr=s0,
            w_ptr=self.W_fused.weight.T.contiguous(),
            b_ptr=self.W_fused.bias,
            out_ptr=out,
            s_out_ptr=s_out,
            B=B, T=T, D=D,
            stride_x_bt=T * D,
            stride_x_bd=D,
            stride_wi=6 * D
        )
        return out, s_out

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
        x = x.contiguous() #.view(-1, self.config.hidden_dim)

        logits = self.output_proj(x)
#        logits = logits.view(B, T, self.config.vocab_size)

        if self.config.return_last_states:
            return logits, (h, s)
        else:
            return logits

class LucyRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, fused_ops=False, layer_norm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fused_ops = fused_ops
        self.layer_norm = layer_norm

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layernorm_in = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.layernorm_r = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.layernorm_z = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.layernorm_h = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

        if self.fused_ops:
            self.W_fused = nn.Linear(hidden_dim, 6 * hidden_dim)
        else:
            self.W_r = nn.Linear(hidden_dim, hidden_dim)
            self.W_z = nn.Linear(hidden_dim, hidden_dim)
            self.W_k = nn.Linear(hidden_dim, hidden_dim)
            self.W_v = nn.Linear(hidden_dim, hidden_dim)
            self.W_h = nn.Linear(hidden_dim, hidden_dim)
            self.W_decay = nn.Linear(hidden_dim, hidden_dim)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.orthogonal_(param)

        if self.layer_norm:
            for layernorm in [self.layernorm_in, self.layernorm_r, self.layernorm_z, self.layernorm_h]:
                nn.init.constant_(layernorm.bias, 0)
                nn.init.constant_(layernorm.weight, 1.0)

    def forward(self, x, h_prev, s_prev, mask=None):
        u = self.layernorm_in(self.input_proj(x))

        if self.fused_ops:
            fused = self.W_fused(u)
            r, z, k, v, h_pre, decay_logits = fused.chunk(6, dim=-1)
            r = torch.sigmoid(self.layernorm_r(r))
            z = torch.sigmoid(self.layernorm_z(z))
            decay = torch.sigmoid(decay_logits)
            s = decay * s_prev + (k * v)
            c = torch.tanh(self.layernorm_h(h_pre + s))
        else:
            r = torch.sigmoid(self.layernorm_r(self.W_r(u)))
            z = torch.sigmoid(self.layernorm_z(self.W_z(u)))
            k = self.W_k(u)
            v = self.W_v(u)
            decay = torch.sigmoid(self.W_decay(u))
            s = decay * s_prev + (k * v)
            c = torch.tanh(self.layernorm_h(self.W_h(u + s)))

        h = (1 - z) * c + z * h_prev

        if mask is not None:
            h = mask * h + (1 - mask) * h_prev
            s = mask * s + (1 - mask) * s_prev

        return h, s

class LucyRNN(nn.Module):
    def __init__(self, config: LucyRNNConfig):
        super().__init__()
        self.config = config

        if self.config.kernel_impl not in ["native", "triton"]:
            raise ValueError("kernel_impl must be either 'native' or 'triton'")

        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer_input_dim = config.input_dim * config.stack_order if i == 0 else config.hidden_dim
            self.layers.append(LucyRNNCell(layer_input_dim, config.hidden_dim, config.fused_ops, config.layer_norm))

        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, hidden_states=None, masks=None):
        batch_size, seq_len, feat_dim = x.size()

        if self.config.stack_order > 1:
            stack = self.config.stack_order
            trim_len = seq_len - (seq_len % stack)
            x = x[:, :trim_len, :]
            x = x.view(batch_size, trim_len // stack, feat_dim * stack)
            if masks is not None:
                masks = masks[:, :trim_len, :].view(batch_size, trim_len // stack, stack).all(dim=-1, keepdim=True)
            seq_len = x.size(1)

        if hidden_states is None:
            h = [torch.zeros(batch_size, self.config.hidden_dim, device=x.device)
                 for _ in range(self.config.num_layers)]
            s = [torch.zeros(batch_size, self.config.hidden_dim, device=x.device)
                 for _ in range(self.config.num_layers)]
        else:
            h, s = hidden_states

        if self.config.is_training:
            input_t = x.clone()

            for l, layer in enumerate(self.layers):
                u = layer.layernorm_in(layer.input_proj(input_t))

                if layer.fused_ops:
                    fused = layer.W_fused(u)
                    _, _, k, v, _, decay_logits = fused.chunk(6, dim=-1)
                else:
                    k = layer.W_k(u)
                    v = layer.W_v(u)

                kv = k * v

                if self.config.decay_mode == "learned":
                    decay = torch.sigmoid(decay_logits) if layer.fused_ops else torch.sigmoid(layer.W_decay(u))
                elif self.config.decay_mode == "prefix_sum":
                    lambda_decay = self.config.lambda_decay
                    time = torch.arange(seq_len, device=x.device).float().unsqueeze(0).unsqueeze(-1)
                    log_decay = -lambda_decay * time  # (1, T, 1)
                    log_decay = log_decay.expand(batch_size, seq_len, self.config.hidden_dim)
                    decay = torch.exp(log_decay)
                else:
                    raise ValueError(f"Unknown decay_mode: {self.config.decay_mode}")

                assert kv.shape == decay.shape, f"kv {kv.shape}, decay {decay.shape} mismatch"

                if self.config.decay_mode == "prefix_sum":
                    # Log-space cumulative scan for numerical stability
                    log_decay = torch.log(decay + 1e-7)           # shape: (B, T, D)
                    log_weights = torch.cumsum(log_decay, dim=1)  # log(cumprod)
                    kv_weighted = kv * torch.exp(log_weights)
                    s_all = torch.cumsum(kv_weighted, dim=1) / (torch.exp(log_weights) + 1e-7)
                elif self.config.kernel_impl == 'triton':
                    s_all = torch.empty_like(kv)
                    B, T, D = kv.shape
                    grid = (B, D)
                    fused_decay_scan[grid](
                        kv_ptr=kv, decay_ptr=decay, output_ptr=s_all,
                        B=B, T=T, D=D,
                        stride_b=T * D, stride_t=D, stride_d=1
                    )
                else:
                    s_t = torch.zeros(batch_size, self.config.hidden_dim, device=x.device)
                    s_all = []
                    for t in range(seq_len):
                        s_t = decay[:, t, :] * s_t + kv[:, t, :]
                        s_all.append(s_t.unsqueeze(1))
                    s_all = torch.cat(s_all, dim=1)

                layer_output = torch.zeros(batch_size, seq_len, self.config.hidden_dim, device=x.device)
                for t in range(seq_len):
                    x_t = input_t[:, t, :]
                    s_t = s_all[:, t, :]
                    mask_t = masks[:, t, :].unsqueeze(-1) if masks is not None else None
                    h[l], _ = layer(x_t, h[l], s_t, mask_t)
                    layer_output[:, t, :] = h[l]

                input_t = layer_output

            outputs = input_t

        else:
            outputs = []
            for t in range(seq_len):
                input_t = x[:, t, :]
                mask_t = masks[:, t, :].unsqueeze(-1) if masks is not None else None

                for l, layer in enumerate(self.layers):
                    h[l], s[l] = layer(input_t, h[l], s[l], mask_t)
                    input_t = h[l]

                outputs.append(h[-1].unsqueeze(1))

            outputs = torch.cat(outputs, dim=1)

        logits = self.output_proj(outputs)

        if self.config.return_last_states:
            return logits, (h, s)
        else:
            return logits

