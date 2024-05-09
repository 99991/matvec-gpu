#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

double sec(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

#define STR(x) #x

const char *source = STR(
__kernel
void matvec(
    __global float *A,
    __global float *x,
    __global float *b,
    int m,
    int n
){
    int i = get_global_id(0);

    if (i >= m) return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++){
        sum += A[i * n + j] * x[j];
    }
    b[i] = sum;
});

cl_context context;
cl_command_queue queue;
cl_device_id device;
cl_program program;
cl_kernel matvec_kernel;

void init_opencl(){
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    char name[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    fprintf(stderr, "OpenCL device: %s\n", name);
    program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    matvec_kernel = clCreateKernel(program, "matvec", NULL);
}

typedef struct Matrix {
    float *values;
    cl_mem buf;
    int m;
    int n;
} Matrix;

void mat_upload(Matrix *mat){
    clEnqueueWriteBuffer(queue, mat->buf, CL_TRUE, 0, mat->m * mat->n * sizeof(float), mat->values, 0, NULL, NULL);
}

void mat_download(Matrix *mat){
    clEnqueueReadBuffer(queue, mat->buf, CL_TRUE, 0, mat->m * mat->n * sizeof(float), mat->values, 0, NULL, NULL);
}

void mat_init(Matrix *mat, int m, int n){
    mat->values = malloc(m * n * sizeof(float));
    mat->buf = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), NULL, NULL);
    mat->m = m;
    mat->n = n;
    for (int i = 0; i < m * n; i++){
        mat->values[i] = rand() / (float)RAND_MAX;
    }
    mat_upload(mat);
}

void mat_free(Matrix *mat){
    free(mat->values);
    clReleaseMemObject(mat->buf);
}

void matvec(Matrix *A, Matrix *x, Matrix *b){
    // Compute b = A x
    cl_kernel kernel = matvec_kernel;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &A->buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &x->buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &b->buf);
    clSetKernelArg(kernel, 3, sizeof(int), &A->m);
    clSetKernelArg(kernel, 4, sizeof(int), &A->n);
    size_t block_size = 4;
    size_t global_size = A->m;
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &block_size, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
}

void matvec_cpu(Matrix *A, Matrix *x, Matrix *b){
    int m = A->m;
    int n = A->n;
    for (int i = 0; i < m; i++){
        float sum = 0.0f;
        for (int j = 0; j < n; j++){
            sum += A->values[i * n + j] * x->values[j];
        }
        b->values[i] = sum;
    }
}

int main(){
    init_opencl();

    int m = 512;
    int n = 1024;

    Matrix A[1], x[1], b[1], b_expected[1];
    mat_init(A, m, n);
    mat_init(x, n, 1);
    mat_init(b, m, 1);
    mat_init(b_expected, m, 1);

    matvec_cpu(A, x, b_expected);

    for (int k = 0; k < 100; k++){
        clFinish(queue);

        double t = sec();

        matvec(A, x, b);
        mat_download(b);
        clFinish(queue);

        double dt = sec() - t;

        printf("%9.3f µs\n", dt * 1e6);

        for (int i = 0; i < m; i++){
            float err = fabsf(b->values[i] - b_expected->values[i]);
            assert(err < 1e-4f);
            assert(b->values[i] > 0.0f);
        }
    }

    mat_free(A);
    mat_free(x);
    mat_free(b);
    mat_free(b_expected);

    clReleaseKernel(matvec_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
