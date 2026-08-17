import triton
import triton.language as tl

# one program process BLOCK elements
# x: [N,]
# o: [N,]
@triton.jit
def silu(x_ptr, o_ptr, N, BLOCK):
	pid = tl.program_id(0)
	cols = pid * BLOCK + tl.arange(0, BLOCK)
	mask = cols < N
	x = tl.load(x_ptr + cols, mask=mask).to(tl.float32)
	tl.store(o_ptr + cols, x * tl.sigmoid(x), mask=mask)

# x1: [N,]
# x2: [N,]
# o: [N,]
@triton.jit
def gilu(x1_ptr, x2_ptr, o_ptr, N, BLOCK):
	pid = tl.program_id(0)
	cols = pid * BLOCK + tl.arange(0, BLOCK)
	mask = cols < N
	x1 = tl.load(x1_ptr + cols, mask=mask).to(tl.float32)
	x2 = tl.load(x2_ptr + cols, mask=mask).to(tl.float32)
	tl.store(o_ptr + cols, x1 * tl.sigmoid(x1) * x2, mask=mask)