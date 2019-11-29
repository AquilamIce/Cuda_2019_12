#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// includes CUDA
#include "cublas_v2.h"
#include <cuda_runtime.h>

#include "Matrix.h"

#define TILE_SIZE 16

double cpuTimer();


void MatrixPrint(Matrix &A);
void MatrixMul(Matrix &prodA, Matrix &prodB, Matrix &resC);
void MatrixCompare(Matrix &A, Matrix &B);
void InitializeMatrix(Matrix &A, int val);
void InitializeMatrix(Matrix &A);

void CublasMultiply(Matrix &A, Matrix &B, Matrix &C);

__global__ void MatrixMulNaive(Matrix A, Matrix B, Matrix C);
__global__ void MatrixMulShared(const Matrix prodA, const Matrix prodB, Matrix resC);
