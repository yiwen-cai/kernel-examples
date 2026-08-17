import triton
import triton.language as tl


# x: [*, N]
# o: [*, N]
@triton.jit
def scan(x_ptr, o_ptr, N, BLOCK):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask)
    tl.store(o_ptr + row * N + cols, tl.cumsum(x, axis=-1), mask=mask)
