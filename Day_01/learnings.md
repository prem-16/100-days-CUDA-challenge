# CUDA Notes

## Chapter 1: Introduction

- **Throughput-oriented design:** Maximizes execution throughput rather than reducing latency.
- **Memory types:** GDDR, SDRAM, and high-bandwidth off-chip memory.
- **CUDA memory transfers:** System to global memory at 4 GB/s (bidirectional).
- **NVLINK:** Supports GPU-GPU interconnects up to 40 GB/s per channel.
- **Parallel execution limits:**
  - Speedup constrained by **Amdahl’s Law**.
  - DRAM bandwidth often limits performance (~10X speedup).
- **CPU-GPU relationship:** GPUs complement CPUs but may introduce overhead.

## Chapter 2: Data Parallel Computing

- **Thread organization:** CUDA uses `threadIdx` and `blockIdx` for parallel execution.
- **SPMD vs SIMD:**
  - **SPMD (Single Program Multiple Data):** Processing units execute the same program but not necessarily in sync.
  - **SIMD (Single Instruction Multiple Data):** All units execute the same instruction at the same time.
- **Threads per block:**
  - CUDA 3.0+ allows up to **1024** threads per block.
  - Older versions allow **512** threads per block.
- **Memory-bound applications:** Performance is often limited by memory access speed rather than compute power.
