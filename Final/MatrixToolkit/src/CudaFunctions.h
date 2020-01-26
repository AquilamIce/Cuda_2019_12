#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "Matrix.h"

#define TILE_SIZE 16

__global__ void MatrixAdd(const Matrix termA, const Matrix termB, Matrix result);

__global__ void MatrixSubtract(const Matrix termA, const Matrix termB, Matrix result);

__global__ void MatrixMultiply( Matrix factA,  Matrix factB, Matrix result);
