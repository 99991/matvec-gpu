from taichi_cuda_matvec import main
import taichi as ti

if __name__ == "__main__":
    ti.init(arch=ti.vulkan)
    main()
