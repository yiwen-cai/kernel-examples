import triton
import triton.language as tl

@triton.jit
def int8_per_group_quant(x_ptr, q_ptr, s_ptr, N, GROUP: tl.constexpr,BLOCK: tl.constexpr):
    row = tl.program_id(0)
    gid = tl.program_id(1)
    cols = tl.arange(0, GROUP)
    mask = cols < GROUP
    x = tl.load(x_ptr + row * N + gid * GROUP + cols, mask=mask, other=0.0).to(tl.float32)
    amax = tl.abs(x).max(axis=0)
    scale = amax / 127.0
    inv = 1.0 / tl.maximum(scale, 1e-8)
    q = tl.extra.cuda.libdevice.rint(x * inv)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    ng = tl.cdiv(N, GROUP)
    tl.store(q_ptr + row * N + gid * GROUP + cols, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + row * ng + gid , scale)
