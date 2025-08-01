import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class LucyRNNConfig:
    input_dim: int
    hidden_dim: int
    num_layers: int
    vocab_size: int
    return_last_states: bool = True
    kernel_impl: str = "native"  # Options: 'native', 'triton'

class LucyRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layernorm_in = nn.LayerNorm(hidden_dim)
        self.layernorm_r = nn.LayerNorm(hidden_dim)
        self.layernorm_z = nn.LayerNorm(hidden_dim)
        self.layernorm_h = nn.LayerNorm(hidden_dim)

        self.W_r = nn.Linear(hidden_dim, hidden_dim)
        self.W_z = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

        self.pos_decay = nn.Parameter(torch.full((hidden_dim,), -0.1))

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
        r = torch.sigmoid(self.layernorm_r(self.W_r(u)))
        z = torch.sigmoid(self.layernorm_z(self.W_z(u)))

        k = self.W_k(u)
        v = self.W_v(u)
        s = (self.pos_decay * s_prev) + (k * v)

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
        
        if self.config.kernel_impl == 'triton':
            raise NotImplementedError("Triton kernels not implemented yet.")

        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer_input_dim = config.input_dim if i == 0 else config.hidden_dim
            self.layers.append(LucyRNNCell(layer_input_dim, config.hidden_dim))

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

# Example instantiation of LucyRNNConfig
lucyrnn_config = LucyRNNConfig(
    input_dim=80,  # match this to your actual input feature dimension
    hidden_dim=512,
    num_layers=4,
    vocab_size=1024,
    return_last_states=True,
    kernel_impl="native",  # use 'triton' when implemented
)

# Instantiate LucyRNN model
lucyrnn = LucyRNN(lucyrnn_config).to("cuda")

