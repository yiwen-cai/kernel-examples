// SM80 BF16 TN GEMM：cp.async + ldmatrix + mma.sync
// A:(M,K) row-major；B:(N,K) row-major（均为 K 连续）；C:(M,N) row-major
// 参考：cutlass/examples/cute/tutorial/sgemm_sm80.cu
#include <cute/tensor.hpp>
using namespace cute;

__global__ void gemm(bfloat16_t const *A, bfloat16_t const *B, bfloat16_t *C, int M, int N, int K) {
	auto mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, K), make_stride(K, _1{})));
	auto mB = make_tensor(make_gmem_ptr(A), make_layout(make_shape(N, K), make_stride(K, _1{})));
	auto mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(M, N), make_stride(N, _1{})));

	auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, make_layout(make_shape(_2{}, _4{}, _1{})), Tile<_32, _32, _16>{});
	auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

	auto bM = Int<128>{};
	auto bN = bM;
	auto bK = Int<64>{};

	auto gA = local_tile(mA, make_shape(bM, bK), make_coord(blockIdx.y, _));
	auto gB = local_tile(mA, make_shape(bN, bK), make_coord(blockIdx.x, _));
	auto gC = local_tile(mC, make_shape(bM, bN), make_coord(blockIdx.y, blockIdx.x));

	auto swizzle_atom = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{});
	auto sa_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK));
	auto sb_layout = tile_to_shape(swizzle_atom, make_shape(bN, bK));
	extern __shared__ char raw[];
	auto sA = make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t *>(raw)), sa_layout);
	auto sB = make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t *>(raw) + cosize(sa_layout)), sb_layout);

	using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, bfloat16_t>;
	auto copyA = make_tiled_copy(CopyAtom{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
	auto copyB = copyA;

	auto tAgA = copyA.get_slice(threadIdx.x).partition_S(gA);
	auto tAsA = copyA.get_slice(threadIdx.x).partition_D(sA);
	auto tBgB = copyB.get_slice(threadIdx.x).partition_S(gB);
	auto tBsB = copyB.get_slice(threadIdx.x).partition_D(sB);

	auto tCgC = thr_mma.partition_C(gC);
	auto tCrA = thr_mma.partition_fragment_A(sA);
	auto tCrB = thr_mma.partition_fragment_A(sB);
	auto tCrC = thr_mma.partition_fragment_C(tCgC);
	clear(tCrC);

	TiledCopy s2r_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, tiled_mma);
	TiledCopy s2r_b = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, bfloat16_t>{}, tiled_mma);
	auto tXsA = s2r_a.get_slice(threadIdx.x).partition_S(sA);
	auto tXrA = s2r_a.get_slice(threadIdx.x).retile_D(tCrA);
	auto tXsB = s2r_b.get_slice(threadIdx.x).partition_S(sB);
	auto tXrB = s2r_b.get_slice(threadIdx.x).retile_D(tCrB);

	for (int k = 0; k < size<2>(gA); k++) {
		copy(copyA, gA(_, _, k), tAsA);
		copy(copyB, gB(_, _, k), tBsB);
		cp_async_fence();
		cp_async_wait<0>();
		__syncthreads();

		copy(s2r_a, tXrA, tXsA);
		copy(s2r_b, tXrB, tXsB);
		gemm(thr_mma, tCrA, tCrB, tCrC);
		__syncthreads();
	}

	for (int i = 0; i < size(tCrC); ++i) {
		tCgC(i) = bfloat16_t(float(tCrC(i)));
	}
}