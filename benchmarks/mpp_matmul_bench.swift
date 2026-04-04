// MPP matmul2d benchmark for Flux2 Klein on M3 Ultra
// Measures TFLOPS for bf16 GEMM at the exact shapes used in our transformer
//
// Usage: swiftc -O -framework Metal -framework Foundation mpp_matmul_bench.swift -o mpp_bench && ./mpp_bench

import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
print("Device: \(device.name)")
print("Metal 4: \(device.supportsFamily(.metal4))")

guard device.supportsFamily(.metal4) else {
    fatalError("Metal 4 required for MPP tensor ops")
}

struct BenchConfig {
    let name: String
    let M: Int
    let N: Int
    let K: Int
}

// Flux2 Klein shapes
let configs = [
    BenchConfig(name: "QKV+MLP proj (single block)", M: 4608, N: 27648, K: 3072),
    BenchConfig(name: "Output proj (single block)", M: 4608, N: 3072, K: 12288),
    BenchConfig(name: "MLP gate only", M: 4608, N: 9216, K: 3072),
]

func createShader(M: Int, N: Int, K: Int, blockM: Int, blockN: Int, blockK: Int, nSimdgroups: Int, useBF16: Bool) -> String {
    let dtype = useBF16 ? "bfloat" : "half"
    return """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void mpp_gemm(
    device \(dtype) *A_buf [[buffer(0)]],
    device \(dtype) *B_buf [[buffer(1)]],
    device \(dtype) *C_buf [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // B is transposed: A[K,M] x B[K,N]^T -> C[N,M]
    auto A = tensor<device \(dtype), dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(\(K), \(M)));
    auto B = tensor<device \(dtype), dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(\(K), \(N)));
    auto C = tensor<device \(dtype), dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(\(N), \(M)));

    constexpr auto desc = matmul2d_descriptor(
        \(blockM), \(blockN), dynamic_length_v<int>,
        false, true, false, matmul2d_descriptor::mode::multiply
    );
    matmul2d<desc, execution_simdgroups<\(nSimdgroups)>> op;

    auto mA = A.slice(0, tgid.y * \(blockM));
    auto mB = B.slice(0, tgid.x * \(blockN));

    // Use cooperative_tensor for accumulation in float32
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] = 0;
    }

    op.run(mA, mB, cT);

    // Cast and store
    auto mC = C.slice(tgid.x * \(blockN), tgid.y * \(blockM));
    auto outT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), \(dtype)>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < outT.get_capacity(); ++i) {
        if (outT.is_valid_element(i))
            outT[i] = \(dtype)(cT[i]);
    }
    outT.store(mC);
}
"""
}

func benchmark(config: BenchConfig) {
    let M = config.M
    let N = config.N
    let K = config.K
    let blockM = 64
    let blockN = 64
    let blockK = 64
    let nSimdgroups = 4

    print("\n\(config.name): (\(M), \(K)) x (\(K), \(N))")
    let flops = 2.0 * Double(M) * Double(N) * Double(K)
    print("  FLOPs: \(String(format: "%.1f", flops / 1e9)) GFLOPS")

    // BF16 version
    let source = createShader(M: M, N: N, K: K, blockM: blockM, blockN: blockN, blockK: blockK, nSimdgroups: nSimdgroups, useBF16: true)

    let library: MTLLibrary
    do {
        library = try device.makeLibrary(source: source, options: nil)
    } catch {
        print("  FAILED to compile: \(error)")
        return
    }

    guard let function = library.makeFunction(name: "mpp_gemm") else {
        print("  FAILED: no function")
        return
    }

    let pso: MTLComputePipelineState
    do {
        pso = try device.makeComputePipelineState(function: function)
    } catch {
        print("  FAILED PSO: \(error)")
        return
    }

    let queue = device.makeCommandQueue()!
    let elemSize = 2  // bf16
    let bufA = device.makeBuffer(length: M * K * elemSize, options: .storageModeShared)!
    let bufB = device.makeBuffer(length: K * N * elemSize, options: .storageModeShared)!
    let bufC = device.makeBuffer(length: M * N * elemSize, options: .storageModeShared)!

    let simdWidth = pso.threadExecutionWidth
    let threadsPerTG = MTLSize(width: simdWidth * nSimdgroups, height: 1, depth: 1)
    let tgCount = MTLSize(width: (N + blockN - 1) / blockN, height: (M + blockM - 1) / blockM, depth: 1)

    print("  Grid: \(tgCount.width)x\(tgCount.height) TGs, \(simdWidth*nSimdgroups) threads each")
    print("  PSO creation: OK")

    // Warmup
    for _ in 0..<5 {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC, offset: 0, index: 2)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: threadsPerTG)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    // Benchmark
    let RUNS = 20
    var times: [Double] = []
    for _ in 0..<RUNS {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufC, offset: 0, index: 2)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: threadsPerTG)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let elapsed = cb.gpuEndTime - cb.gpuStartTime
        times.append(elapsed)
    }

    let minTime = times.min()!
    let avgTime = times.reduce(0, +) / Double(RUNS)
    let tflops = flops / minTime / 1e12

    print("  BF16 MPP matmul2d:")
    print("    Min: \(String(format: "%.2f", minTime * 1000)) ms")
    print("    Avg: \(String(format: "%.2f", avgTime * 1000)) ms")
    print("    TFLOPS: \(String(format: "%.2f", tflops))")
    print("    Utilization: \(String(format: "%.0f", tflops / 27 * 100))% of M3 Ultra peak (27T)")
}

print("\n" + String(repeating: "=", count: 60))
print("MPP matmul2d Benchmark — Flux2 Klein shapes on M3 Ultra")
print(String(repeating: "=", count: 60))

for config in configs {
    benchmark(config: config)
}

print("\nDone. Compare these TFLOPS with MLX Steel GEMM to see if MPP is faster.")
