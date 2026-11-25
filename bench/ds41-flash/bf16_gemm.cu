#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

// Event-timed dense BF16-input GEMM with FP32 accumulation and output.
static void ck(cudaError_t e, const char *where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e)); std::exit(2); }
}
static void cb(cublasStatus_t e, const char *where) {
    if (e != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: cuBLAS status %d\n", where, int(e)); std::exit(2); }
}

int main() {
    constexpr int n = 8192;
    constexpr int warmup = 5;
    constexpr int iterations = 20;
    cublasHandle_t h = nullptr;
    cudaStream_t stream = nullptr;
    __nv_bfloat16 *a = nullptr, *b = nullptr;
    float *c = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cb(cublasCreate(&h), "cublasCreate");
    cb(cublasSetStream(h, stream), "cublasSetStream");
    ck(cudaMalloc(&a, std::size_t(n) * n * sizeof(*a)), "cudaMalloc(a)");
    ck(cudaMalloc(&b, std::size_t(n) * n * sizeof(*b)), "cudaMalloc(b)");
    ck(cudaMalloc(&c, std::size_t(n) * n * sizeof(*c)), "cudaMalloc(c)");
    ck(cudaMemset(a, 0, std::size_t(n) * n * sizeof(*a)), "cudaMemset(a)");
    ck(cudaMemset(b, 0, std::size_t(n) * n * sizeof(*b)), "cudaMemset(b)");
    ck(cudaMemset(c, 0, std::size_t(n) * n * sizeof(*c)), "cudaMemset(c)");
    ck(cudaEventCreate(&start), "cudaEventCreate(start)");
    ck(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    const float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < warmup; ++i)
        cb(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a,
                     CUDA_R_16BF, n, b, CUDA_R_16BF, n, &beta, c, CUDA_R_32F,
                     n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "warmup GEMM");
    ck(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i)
        cb(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a,
                     CUDA_R_16BF, n, b, CUDA_R_16BF, n, &beta, c, CUDA_R_32F,
                     n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "timed GEMM");
    ck(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    ck(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    std::printf("BF16 GEMM %dx%dx%d: %.3f ms/iter, %.2f TFLOP/s\n", n, n, n,
                ms / iterations, (2.0 * n * n * n * iterations) / (ms * 1e9));
    cudaFree(a); cudaFree(b); cudaFree(c);
    cudaEventDestroy(start); cudaEventDestroy(stop); cublasDestroy(h);
    return 0;
}
