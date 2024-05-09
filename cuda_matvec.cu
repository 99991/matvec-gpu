#include <time.h>
#include <assert.h>
#include <stdio.h>

double sec(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

__global__
void matvec(float *A, float *x, float *b, int m, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= m) return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++){
        sum += A[i * n + j] * x[j];
    }
    b[i] = sum;
}

void init(float *values, int n){
    for (int i = 0; i < n; i++){
        values[i] = rand() / (float)RAND_MAX;
    }
}

void matvec_cpu(float *A, float *x, float *b, int m, int n){
    for (int i = 0; i < m; i++){
        float sum = 0.0f;
        for (int j = 0; j < n; j++){
            sum += A[i * n + j] * x[j];
        }
        b[i] = sum;
    }
}

int main() {
    int m = 512;
    int n = 1024;

    float *A, *x, *b;
    cudaMalloc((void **)&A, m * n * sizeof(float));
    cudaMalloc((void **)&x, n * sizeof(float));
    cudaMalloc((void **)&b, m * sizeof(float));
    cudaDeviceSynchronize();
    float *h_A, *h_x, *h_b, *h_b_expected;
    cudaMallocHost((void **)&h_A, m * n * sizeof(float));
    cudaMallocHost((void **)&h_x, n * sizeof(float));
    cudaMallocHost((void **)&h_b, m * sizeof(float));
    cudaMallocHost((void **)&h_b_expected, m * sizeof(float));
    init(h_A, m * n);
    init(h_x, n);
    init(h_b, m);
    matvec_cpu(h_A, h_x, h_b_expected, m, n);
    cudaMemcpy(A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int k = 0; k < 100; k++){
        cudaDeviceSynchronize();

        double t = sec();

        int block_size = 8;
        int num_blocks = (m + block_size - 1) / block_size;
        matvec<<<num_blocks, block_size>>>(A, x, b, m, n);
        cudaMemcpy(h_b, b, m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        double dt = sec() - t;

        printf("%9.3f µs\n", dt * 1e6);

        for (int i = 0; i < m; i++){
            float err = fabsf(h_b[i] - h_b_expected[i]);
            assert(err < 1e-4f);
            assert(h_b[i] > 0.0f);
        }
    }

    cudaFree(A);
    cudaFree(x);
    cudaFree(b);
    cudaFreeHost(h_A);
    cudaFreeHost(h_x);
    cudaFreeHost(h_b);
    cudaFreeHost(h_b_expected);

    return 0;
}
