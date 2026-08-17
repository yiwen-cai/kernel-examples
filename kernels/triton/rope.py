import triton
import triton.language as tl

@triton.jit
def rope(x_ptr, o_ptr, cos_ptr, sin_ptr, S, D: tl.constexpr):
	pid = tl.program_id(0)
	offsets = tl.arange(0, D // 2)
	s = pid % S

	x1 = tl.load(x_ptr + pid * D + offsets * 2)          # rotate_half: pid * D + offsets
	x2 = tl.load(x_ptr + pid * D + offsets * 2 + 1)      # rotate_half: pid * D + offsets + D // 2
	sin = tl.load(sin_ptr + s * (D // 2) + offsets)
	cos = tl.load(cos_ptr + s * (D // 2) + offsets)

	tl.store(o_ptr + pid * D + offsets * 2, x1 * cos - x2 * sin)          # rotate_half: pid * D + offsets
	tl.store(o_ptr + pid * D + offsets * 2 + 1, x1 * sin + x2 * cos)      # rotate_half: pid * D + offsets + D // 2