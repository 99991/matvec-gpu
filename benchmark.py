import re
import sys
import json
import subprocess
import os

def run(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise Exception(f"Failed to run {' '.join(command)}: {stderr.decode('utf-8')}")

    text = stdout.decode("utf-8")

    times = [float(t.split()[0]) for t in re.findall(r"\d+\.\d+ µs", text)]

    return times

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    run(["nvcc", "-O3", "cuda_matvec.cu", "-o", "cuda_matvec"])
    run(["gcc", "-O3", "opencl_matvec.c", "-o", "opencl_matvec", "-lOpenCL"])
    results = {
        "opencl_matvec.c": run(["./opencl_matvec"]),
        "cuda_matvec.cu": run(["./cuda_matvec"]),
        "cupy_matvec.py": run([sys.executable, "cupy_matvec.py"]),
        "cublas_matvec.py": run([sys.executable, "cublas_matvec.py"]),
        "numba_matvec.py": run([sys.executable, "numba_matvec.py"]),
        "taichi_cuda_matvec.py": run([sys.executable, "taichi_cuda_matvec.py"]),
        "taichi_opengl_matvec.py": run([sys.executable, "taichi_opengl_matvec.py"]),
        "taichi_vulkan_matvec.py": run([sys.executable, "taichi_vulkan_matvec.py"]),
    }
    with open("data/results.json", "w") as f:
        json.dump(results, f, indent=4)
