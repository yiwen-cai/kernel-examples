import triton
import triton.language as tl

@triton.jit
def int8_per_token_quant(x_ptr, q_ptr, s_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    amax = tl.abs(x).max(axis=0)
    scale = amax / 127.0
    inv = 1.0 / tl.maximum(scale, 1e-8)
    q = tl.extra.cuda.libdevice.rint(x * inv)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(q_ptr + row * N + cols, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, scale)
