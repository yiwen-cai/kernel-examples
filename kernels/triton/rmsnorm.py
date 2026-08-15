import triton
import triton.language as tl


@triton.jit
def rmsnorm(x_ptr, g_ptr, o_ptr, eps, N, BLOCK):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + eps)
    g = tl.load(g_ptr + cols, mask=mask)

    tl.store(o_ptr + row * N + cols, x * g * rrms, mask=mask)
