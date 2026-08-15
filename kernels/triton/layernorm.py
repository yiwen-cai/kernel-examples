import triton
import triton.language as tl

@triton.jit
def layerNorm(x_ptr, w_ptr, b_ptr, o_ptr, N, eps, BLOCK):
	row = tl.program_id(0)
	cols = tl.arange(0, BLOCK)
	mask = cols < N
	x = tl.load(x_ptr + row * N + cols, mask=mask, padding_option="zero").to(tl.float32)
	w = tl.load(w_ptr + cols, mask=mask)
	b = tl.load(b_ptr+ cols, mask=mask)
	mean = tl.sum(x, axis=-1) / N
	diff = tl.where(mask, x - mean, 0.0)
	var = tl.sum(diff * diff, axis=-1) / N
	rstd = 1.0 / tl.sqrt(var + eps)
	tl.store(o_ptr + row * N + cols, diff*rstd*w + b, mask=mask)