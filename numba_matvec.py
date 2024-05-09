import time
import numpy as np
import numba
from numba import cuda

m = 512
n = 1024

A = np.random.rand(m, n).astype(np.float32)
x = np.random.rand(n).astype(np.float32)
b = np.random.rand(m).astype(np.float32)

b_expected_np = A @ x
b_np = np.zeros(m, dtype=np.float32)

A = cuda.to_device(A)
x = cuda.to_device(x)
b = cuda.to_device(b)

@cuda.jit
def matvec(A, x, b, m, n):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= m: return
    s = 0.0
    for j in range(n):
        s += A[i, j] * x[j]
    b[i] = s

for _ in range(100):
    cuda.synchronize()
    start_time = time.time()

    block_size = 8
    num_blocks = (m + block_size - 1) // block_size
    matvec[num_blocks, block_size](A, x, b, m, n)
    b.copy_to_host(b_np)

    cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"{elapsed_time * 1e6:9.3f} µs")
    assert np.allclose(b_expected_np, b_np)
