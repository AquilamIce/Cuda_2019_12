#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "Matrix.h"

#define TILE_SIZE 16
#define BLOCK_SIZE 16

__global__ void MatrixAdd(const float *termA, const float *termB,  float *result, const int A, const int B);

__global__ void MatrixSubtract(const float *termA, const float *termB,  float *result, const int A, const int B);

__global__ void MatrixMultiply(const float *termA, const float *termB,  float *result, const int A, const int B, const int C, const int D,  int E,  int F);

__global__ void MatrixTranspose(const float *in, float *out, const int A, const int B, int C);

__global__ void VectorSum(Matrix in, double *out);
