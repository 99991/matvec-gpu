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

for i in range(100):
    cp.cuda.Device().synchronize()

    start_time = time.perf_counter()

    cp.dot(A, x, out=b)

    cp.asnumpy(b, out=b_np)

    cp.cuda.Device().synchronize()

    elapsed_time = time.perf_counter() - start_time

    assert np.allclose(b_expected_np, b_np)

    print(f"{elapsed_time * 1e6:9.3f} µs")
