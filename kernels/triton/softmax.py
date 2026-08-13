import triton
import triton.language as tl

@triton.jit
def softmax(x_ptr, o_ptr, N, BLOCK):
    row = tl.program_id(0)
    m = float('-inf')
    l = 0.0
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=float('-inf'))
        m_new = tl.maximum(m, tl.max(x, axis=0))
        scale = tl.exp(m - m_new)
        l = l * scale + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=float('-inf'))
        tl.store(o_ptr + row * N + cols, tl.exp(x-m) / l, mask=mask)