// includes, system
#include <stdlib.h>
#include <stdio.h>


// includes CUDA
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
//Low accuracy in reduction algorithm for float, no idea why
#define VectorType double
//Calculate sum of vector elements using CPU
VectorType TotalCPU(VectorType *input, int size);
//Initialize vector with random values
void InitializeVector(VectorType *input, int size);


//Kernels
//Calculate sum of vector elements using reduction algorithm
__global__ void ReductionTotalKernel(VectorType *input, VectorType *output, int size);
//Calculate sum of vector elements using naive algorithm
__global__ void NaiveTotalKernel(VectorType *input, VectorType *output, int size);
