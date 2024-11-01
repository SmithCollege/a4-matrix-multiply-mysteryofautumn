#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <sys/time.h>

#define SIZE 10

double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { printf("get time ofday error"); }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main(void) {


    float* M, *N, *P;
    cudaMallocManaged(&M, sizeof(float) * SIZE * SIZE);
    cudaMallocManaged(&N, sizeof(float) * SIZE * SIZE);
    cudaMallocManaged(&P, sizeof(float) * SIZE * SIZE);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            M[i * SIZE + j] = 1; // x[i][j]
            N[i * SIZE + j] = 1;
        }
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f; 
    float beta = 0.0f; 

    double t0 = get_clock();

    cublasSgemm(handle, 
        CUBLAS_OP_N, 
        CUBLAS_OP_N,
        SIZE, 
        SIZE, 
        SIZE,
        &alpha,
        M, SIZE,
        N, SIZE,
        &beta,
        P, SIZE);

    cudaDeviceSynchronize();

    double t1 = get_clock();

    printf("size: %d \n", SIZE);
    printf("time: %f ns\n", (1000000000.0 * (t1 - t0)));



    // Error checking 
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
        if (P[i * SIZE + j] != SIZE) {
            printf("Error at P[%d][%d]: %f\n", i, j, P[i * SIZE + j]);
        }
        }
    }


    cudaFree(N);
    cudaFree(M);
    cudaFree(P);
    
    cublasDestroy(handle);

    return 0;
}
