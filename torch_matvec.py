import torch
import numpy as np
import time

device = torch.device("cuda")

m = 512
n = 1024

A = torch.randn(m, n, device=device)
x = torch.randn(n, device=device)
b = torch.randn(m, device=device)

b_expected_np = (A @ x).cpu().numpy()

for _ in range(100):
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    b = torch.matmul(A, x)
    b_np = b.cpu().numpy()

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    print(f"{elapsed_time * 1e6:9.3f} µs")

    assert np.allclose(b_expected_np, b_np)
