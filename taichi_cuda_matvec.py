import taichi as ti
import numpy as np
import time

@ti.kernel
def init(x: ti.template()):
    for i in ti.grouped(x):
        x[i] = ti.random(ti.float32)

@ti.kernel
def matvec(A: ti.template(), x: ti.template(), b: ti.template()):
    m, n = A.shape
    ti.loop_config(block_dim=8)
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        b[i] = s

def main():
    m = 512
    n = 1024

    A = ti.field(shape=(m, n), dtype=ti.float32)
    x = ti.field(shape=n, dtype=ti.float32)
    b = ti.field(shape=m, dtype=ti.float32)

    init(A)
    init(x)
    init(b)

    b_expected_np = A.to_numpy() @ x.to_numpy()

    for _ in range(100):
        ti.sync()
        start_time = time.perf_counter()

        matvec(A, x, b)

        b_np = b.to_numpy()

        ti.sync()
        elapsed_time = time.perf_counter() - start_time

        print(f"{elapsed_time * 1e6:9.3f} µs")

        assert np.allclose(b_expected_np, b_np)

if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    main()
