#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vecADDKernel(float *A, float *B, float *C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("No CUDA-compatible GPU found!\n");
    }
    else
    {
        printf("CUDA device count: %d\n", deviceCount);
    }
    const int N = 2 << 20;              // elements in vec
    const int size = N * sizeof(float); // total size of vectors in bytes

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // initialize with values in host
    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1;
        h_B[i] = 1;
        h_C[i] = 0;
    }

    float *d_A, *d_B, *d_C;
    // Allocate device memory for d_A and d_B
    cudaMalloc((void **)&d_A, size);
    // copy h_A host to d_A device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_C, size);

    // invoke kernel code
    vecADDKernel<<<ceil(N / 256.0), 256>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // free device memory for A,B, C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceSynchronize();
    // print first 10 element
    for (int i = 0; i < 10; i++)
    {
        printf("array C[%d] =%f\n", i, h_C[i]);
    }
    return 0;
}