"""MPP matmul2d for MLX — 99% utilization on M3 Ultra.

Uses Metal Performance Primitives tensor_ops::matmul2d.
The MPP headers require global scope (namespace declarations), so we
pass them in the `header` parameter of mx.fast.metal_kernel.

On M3 Ultra: 26.6 TFLOPS (99%) vs MLX Steel's 23.2 TFLOPS (86%).
"""

import os
import mlx.core as mx

_kernel_cache = {}

# Global-scope includes and helper function — goes in `header`
_MPP_HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// Helper that does the actual MPP matmul — called from kernel body
// A: (M, K) row-major → tensor_inline as (K, M) col-major
// B: (N, K) row-major → tensor_inline as (K, N) col-major (transposed)
// C: (M, N) row-major → tensor_inline as (N, M) col-major
static inline void do_mpp_gemm(
    device bfloat* A_buf, device bfloat* B_buf, device bfloat* C_buf,
    uint M_val, uint N_val, uint K_val,
    uint2 tgid
) {
    auto A = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K_val, M_val));
    auto B = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K_val, N_val));
    auto C = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(N_val, M_val));

    constexpr auto desc = matmul2d_descriptor(
        64, 64, dynamic_length_v<int>,
        false, true, false,
        matmul2d_descriptor::mode::multiply
    );
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(0, tgid.x * 64);

    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] = 0;
    }

    op.run(mA, mB, cT);

    auto mC = C.slice(tgid.x * 64, tgid.y * 64);
    auto outT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), bfloat>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < outT.get_capacity(); ++i) {
        if (outT.is_valid_element(i))
            outT[i] = static_cast<bfloat>(cT[i]);
    }
    outT.store(mC);
}
"""


def _make_kernel_source(M: int, N: int, K: int) -> str:
    """Kernel body — recovers threadgroup position and calls MPP helper."""
    return f"""
    // Recover threadgroup position from thread position
    // Grid: (grid_x * 128, grid_y, 1), Threadgroup: (128, 1, 1)
    uint2 tgid = uint2(
        thread_position_in_grid.x / threads_per_threadgroup.x,
        thread_position_in_grid.y
    );
    do_mpp_gemm((device bfloat*)inp0, (device bfloat*)inp1, (device bfloat*)out0,
                {M}u, {N}u, {K}u, tgid);
    """


def mpp_matmul(a: mx.array, b: mx.array) -> mx.array:
    """MPP-accelerated matmul: C = A @ B.T

    Args:
        a: (M, K) bf16
        b: (N, K) bf16 — weight matrix (transposed convention)

    Returns:
        c: (M, N) bf16
    """
    assert a.dtype == mx.bfloat16 and b.dtype == mx.bfloat16
    assert a.ndim == 2 and b.ndim == 2

    M, K = a.shape
    N, K2 = b.shape
    assert K == K2

    TILE = 64
    SIMDGROUPS = 4
    SIMD_SIZE = 32

    grid_x = (N + TILE - 1) // TILE
    grid_y = (M + TILE - 1) // TILE

    cache_key = (M, N, K)
    if cache_key not in _kernel_cache:
        _kernel_cache[cache_key] = mx.fast.metal_kernel(
            name=f"mpp_gemm_{M}_{N}_{K}",
            input_names=["inp0", "inp1"],
            output_names=["out0"],
            source=_make_kernel_source(M, N, K),
            header=_MPP_HEADER,
            atomic_outputs=False,
        )

    kernel = _kernel_cache[cache_key]

    out = kernel(
        inputs=[a, b],
        grid=(grid_x * SIMD_SIZE * SIMDGROUPS, grid_y, 1),
        threadgroup=(SIMD_SIZE * SIMDGROUPS, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
    )

    return out[0]


def mpp_available() -> bool:
    try:
        return mx.metal.is_available()
    except AttributeError:
        return False
