#include <iostream>
#include <math.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

#define SIZE 1000
#define BLOCK_SIZE 256
#define TILE_WIDTH 5

double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { printf("get time ofday error"); }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__global__ void MatrixMulKernal(float* M, float* N, float* P,int Width){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if((row<Width)&&(col<Width)){
        float Pvalue = 0;

        for (int i = 0; i < Width; ++i){
            Pvalue += M[row*Width+i] * N[i*Width+col]; 
            P[row*Width+col] = Pvalue;
        }

    }
}

int main(void)
{
    float* M, *N, *P;
    cudaMallocManaged(&M, sizeof(float) * SIZE * SIZE);
    cudaMallocManaged(&N, sizeof(float) * SIZE * SIZE);
    cudaMallocManaged(&P, sizeof(float) * SIZE * SIZE);

    // initialization
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            M[i * SIZE + j] = 1; // x[i][j]
            N[i * SIZE + j] = 1;
        }
    }

    double t0 = get_clock();

    // Run kernel 
    dim3 dimGrid(ceil((1.0*SIZE)/TILE_WIDTH),ceil((1.0*SIZE)/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    MatrixMulKernal<<<dimGrid, dimBlock>>>(M, N, P, SIZE);
  
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    double t1 = get_clock();
    printf("size: %d\n", SIZE);
    printf("time: %f ns\n", (1000000000.0 * (t1 - t0)));


    // Error checking 
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
        if (P[i * SIZE + j] != SIZE) {
            printf("Error at P[%d][%d]: %f\n", i, j, P[i * SIZE + j]);
        }
        }
    }

  // Free memory
    cudaFree(M);
    cudaFree(N);
    cudaFree(P);
  
    return 0;
}