import cupy as cp
import time
import numpy as np

m = 512
n = 1024

A = cp.random.rand(m, n, dtype=cp.float32)
x = cp.random.rand(n, dtype=cp.float32)
b = cp.random.rand(m, dtype=cp.float32)

b_expected_np = cp.asnumpy(A @ x)
b_np = np.zeros(m, dtype=np.float32)

kernel = cp.RawKernel(r"""
extern "C" __global__
void matvec(
    float *A,
    float *x,
    float *b,
    int m,
    int n
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= 1024) return;

    float sum = 0;
    for (int j = 0; j < n; j++){
        sum += A[i * n + j] * x[j];
    }
    b[i] = sum;
}
""", "matvec")

for i in range(100):
    cp.cuda.Device().synchronize()

    start_time = time.perf_counter()

    block_size = 8
    num_blocks = (m + block_size - 1) // block_size
    kernel((num_blocks,), (block_size,), (A, x, b, m, n))

    cp.asnumpy(b, out=b_np)

    cp.cuda.Device().synchronize()

    elapsed_time = time.perf_counter() - start_time

    assert np.allclose(b_expected_np, b_np)

    print(f"{elapsed_time * 1e6:9.3f} µs")
