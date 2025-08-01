import torch
import torch.nn as nn
from dataclasses import dataclass
from lucyrnn_triton import fused_decay_scan

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

class LucyRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, fused_ops=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fused_ops = fused_ops

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layernorm_in = nn.LayerNorm(hidden_dim)
        self.layernorm_r = nn.LayerNorm(hidden_dim)
        self.layernorm_z = nn.LayerNorm(hidden_dim)
        self.layernorm_h = nn.LayerNorm(hidden_dim)

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
            layer_input_dim = config.input_dim if i == 0 else config.hidden_dim
            self.layers.append(LucyRNNCell(layer_input_dim, config.hidden_dim, config.fused_ops))

        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, hidden_states=None, masks=None):
        batch_size, seq_len, _ = x.size()

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
                    decay = torch.sigmoid(decay_logits)
                else:
                    k = layer.W_k(u)
                    v = layer.W_v(u)
                    decay = torch.sigmoid(layer.W_decay(u))

                kv = k * v

                if self.config.kernel_impl == 'triton':
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

                input_t = layer_output  # for next layer

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

