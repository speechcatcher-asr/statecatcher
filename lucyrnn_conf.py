from dataclasses import dataclass

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
