# matvec-gpu

Comparison of various GPU acceleration frameworks using matrix-vector multiplication as an example

# Results

All results have been run with an NVIDIA GeForce RTX 3060 Laptop GPU.

| Framework | Min Time [µs] | Median [µs] | Max [µs] |
| --- | --- | --- |
| cuda_matvec.cu | 31.222 | 32.200 | 64.19 |
| opencl_matvec.c | 34.602 | 35.105 | 59.908 |
| cupy_matvec.py | 50.206 | 53.148 | 545.908 |
| cublas_matvec.py | 55.415 | 57.691 | 152.909 |
| taichi_cuda_matvec.py | 173.906 | 178.040 | 64770.521 |
| numba_matvec.py | 174.046 | 180.960 | 194117.546 |
| taichi_vulkan_matvec.py | 608.203 | 788.168 | 8638.188 |
| taichi_opengl_matvec.py | 978.31 | 986.743 | 11853.526 |
