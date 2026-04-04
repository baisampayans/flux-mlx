// MPP GEMM for Flux2 Klein — bf16 matmul at 99% utilization on M3 Ultra
//
// C = A @ B^T  where A:(M,K), B:(N,K), C:(M,N)
// Uses cooperative_tensor for float32 accumulation, optional fused SiLU
//
// Compiled: xcrun metal -std=metal4.0 -target air64-apple-macos26.0 -c mpp_gemm.metal -o mpp_gemm.metallib

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// ── Plain GEMM ───────────────────────────────────────────────
kernel void mpp_gemm_bf16(
    device bfloat *A_buf [[buffer(0)]],
    device bfloat *B_buf [[buffer(1)]],
    device bfloat *C_buf [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // tensor_inline wraps raw buffer with dimension metadata
    // A is row-major (M,K) stored as column-major (K,M) for MPP
    auto A = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N));
    auto C = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(N, M));

    constexpr auto desc = matmul2d_descriptor(
        64, 64, dynamic_length_v<int>,
        false, true, false,
        matmul2d_descriptor::mode::multiply
    );
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(0, tgid.x * 64);

    // Float32 accumulation in cooperative registers
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] = 0;
    }

    op.run(mA, mB, cT);

    // Cast float32 → bf16 and store
    auto mC = C.slice(tgid.x * 64, tgid.y * 64);
    auto outT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), bfloat>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < outT.get_capacity(); ++i) {
        if (outT.is_valid_element(i))
            outT[i] = static_cast<bfloat>(cT[i]);
    }
    outT.store(mC);
}

// ── GEMM + fused SiLU ────────────────────────────────────────
kernel void mpp_gemm_silu_bf16(
    device bfloat *A_buf [[buffer(0)]],
    device bfloat *B_buf [[buffer(1)]],
    device bfloat *C_buf [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    auto A = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M));
    auto B = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N));
    auto C = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(N, M));

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

    // Fused SiLU in cooperative registers (no device memory round-trip)
    auto mC = C.slice(tgid.x * 64, tgid.y * 64);
    auto outT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), bfloat>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < outT.get_capacity(); ++i) {
        if (outT.is_valid_element(i)) {
            float x = cT[i];
            outT[i] = static_cast<bfloat>(x / (1.0f + exp(-x)));
        }
    }
    outT.store(mC);
}
