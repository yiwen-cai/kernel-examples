import triton
import triton.language as tl

@triton.jit
def flashattention(q_ptr, k_ptr, v_ptr, o_ptr, sqh, skh, svh, soh, S, scale, BQ, BK, D):
	pid_h = tl.program_id(0)
	pid_q = tl.program_id(1)
	q_base = q_ptr + pid_h * sqh
	k_base = k_ptr + pid_h * skh
	v_base = v_ptr + pid_h * svh
	o_base = o_ptr + pid_h * soh
	q_blk = tl.make_block_ptr(base=q_base, shape=(S,D), strides=(D, 1), offsets=(pid_q * BQ,0), block_shape=(BQ, D), order=(1, 0))
	k_blk = tl.make_block_ptr(base=k_base, shape=(S,D), strides=(D, 1), offsets=(0,0), block_shape=(BK, D), order=(1, 0))
	v_blk = tl.make_block_ptr(base=v_base, shape=(S,D), strides=(D, 1), offsets=(0,0), block_shape=(BK, D), order=(1, 0))
	o_blk = tl.make_block_ptr(base=o_base, shape=(S,D), strides=(D, 1), offsets=(pid_q * BQ,0), block_shape=(BQ, D), order=(1, 0))

	q = tl.load(q_blk, boundary_check=(0,), padding_option="zero")
	m_i = tl.full(shape=[BQ], value=float('-inf'), dtype=tl.float32)
	l_i = tl.zeros(shape=[BQ])
	o_i = tl.zeros(shape=[BQ])
	q_off = pid_q * sqh + tl.arange(0, BQ)
	for k0 in range(0, S, BK):
		k_off = tl.arange(k0, k0+BK)
		k = tl.load(k_blk, boundary_check=(0,), padding_option="zero")
		s= tl.dot(q_blk, tl.trans(k)) * scale
		s += tl.where(q_off >= k_off, 0, -1e6)
		m_new = tl.maximum(m_i, tl.max(s, axis=1))
		p = tl.exp(s - m_new[:, None])
		alpha = tl.exp(m_i-m_new)
		l_i = l_i * alpha + tl.sum(p, axis=1)
		v = tl.load(v_blk, boundary_check=(0,), padding_option="zero")
		o_i = o_i * alpha + tl.dot(p.to(v.dtype), v)
		m_i = m_new
		k_blk = tl.advance(k_blk, offsets=(BK, 0))
		v_blk = tl.advance(v_blk, offsets=(BK, 0))
	tl.store(o_blk, (o_i / l_i[:, None]).to(tl.bfloat16), boundary_check=(0,))