#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "Matrix.h" // Header file of Matrix class

#define TILE_SIZE 16 //Definition of any Tile size
#define BLOCK_SIZE 16 //Definition of Block size

//Matrix addition kernel
__global__ void MatrixAdd(const float *termA, const float *termB,  float *result, const int A, const int B);

//Matrix substraction kernel
__global__ void MatrixSubtract(const float *termA, const float *termB,  float *result, const int A, const int B);

//Matrix multiplication kernel
__global__ void MatrixMultiply(const float *termA, const float *termB,  float *result, const int A, const int B, const int C, const int D,  int E,  int F);

//Matrix transposition kernel
__global__ void MatrixTranspose(const float *in, float *out, const int A, const int B, int C);

//Matrix elements summation kernel
__global__ void VectorSum(Matrix in, double *out);
