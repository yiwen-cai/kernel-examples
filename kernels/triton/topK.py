import triton
import triton.language as tl

# x: [*, N]
# v: [*, K]
# i: [*, K]
@triton.jit
def topK(x_ptr, v_ptr, i_ptr, N, K:tl.constexpr, BLOCK: tl.constexpr):
	row = tl.program_id(0)
	cols = tl.arange(0, BLOCK)

	x = tl.load(x_ptr + row * N + cols, mask=cols < N, other=float('-inf'))
	for k in range(K):
		idx = tl.argmax(x, axis=0)
		tl.store(v_ptr + row * K + k, tl.max(x, axis=0))
		tl.store(i_ptr + row * K + k, idx)
		x = tl.where(cols == idx, float('-inf'), x)
