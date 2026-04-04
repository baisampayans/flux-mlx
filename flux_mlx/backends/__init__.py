"""Backend selection for Flux2 transformer matmul acceleration.

auto: Standard MLX ops (all Apple Silicon)
mpp:  Metal Performance Primitives matmul2d (macOS 26+, 99% GPU utilization)
"""
