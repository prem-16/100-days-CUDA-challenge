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
- **CUDA MALLOC:** The first parameter to the cudaMalloc function is the address of a pointer variable that will be set to point to the allocated object. The address of the pointer variable should be cast to (void **) because the function expects a generic pointer; the memory allocation function is a generic function that is not restricted to any particular type of objects.2 This parameter allows the cudaMalloc function to write the address of the allocated memory into the pointer variable.3 The host code to launch kernels passes this pointer value to the kernels that need to access the allocated memory object. The second parameter to the cudaMalloc function gives the size of the data to be allocated, in number of bytes. The usage of this second parameter is consistent with the size parameter to the C malloc function.
