import triton
import triton.language as tl
import torch

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
        # ----------------------------------------------------------

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


@triton.jit
def rnn_forward_unfused(
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
        # Load all gates
        r = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 0 * stride_g_cd + d)
        z = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 1 * stride_g_cd + d)
        k = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 2 * stride_g_cd + d)
        v = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 3 * stride_g_cd + d)
        h_pre = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 4 * stride_g_cd + d)
        decay = tl.load(gates_ptr + b * stride_g_bt + t * stride_g_td + 5 * stride_g_cd + d)

        # Apply non-linearities
        r = tl.sigmoid(r)
        z = tl.sigmoid(z)
        decay = tl.sigmoid(decay)

        kv = k * v
        s = decay * s + kv
        c = tl.sigmoid(2 * (h_pre + s)) * 2.0 - 1.0
        h = (1. - z) * c + z * h

        tl.store(out_ptr + b * stride_o_bt + t * stride_o_bd + d, h)

    tl.store(s_out_ptr + b * D + d, s)

# This one doesnt work yet and throws a CUDA error
@triton.jit
def fused_rnn_forward(
    x_ptr, h0_ptr, s0_ptr, w_ptr, b_ptr, out_ptr, s_out_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    stride_x_bt: tl.constexpr, stride_x_bd: tl.constexpr,
    stride_wi: tl.constexpr,  # should now be D, not 6D
):
    b = tl.program_id(0)
    d = tl.program_id(1)

    if b >= B or d >= D:
        return

    h = tl.load(h0_ptr + b * D + d)
    s = tl.load(s0_ptr + b * D + d)

    for t in range(T):
        x = tl.load(x_ptr + b * stride_x_bt + t * stride_x_bd + d)

        r = x * tl.load(w_ptr + d + D * 0) + tl.load(b_ptr + d + D * 0)
        z = x * tl.load(w_ptr + d + D * 1) + tl.load(b_ptr + d + D * 1)
        k = x * tl.load(w_ptr + d + D * 2) + tl.load(b_ptr + d + D * 2)
        v = x * tl.load(w_ptr + d + D * 3) + tl.load(b_ptr + d + D * 3)
        h_pre = x * tl.load(w_ptr + d + D * 4) + tl.load(b_ptr + d + D * 4)
        decay = tl.sigmoid(x * tl.load(w_ptr + d + D * 5) + tl.load(b_ptr + d + D * 5))

        r = tl.sigmoid(r)
        z = tl.sigmoid(z)

        kv = k * v
        s = decay * s + kv
        c = tl.sigmoid(2 * (h_pre + s)) * 2.0 - 1.0
        h = (1. - z) * c + z * h

        tl.store(out_ptr + b * stride_x_bt + t * stride_x_bd + d, h)

    tl.store(s_out_ptr + b * D + d, s)


