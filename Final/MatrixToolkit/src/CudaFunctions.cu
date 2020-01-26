#include "CudaFunctions.h"

#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "Matrix.h"

__global__ void MatrixAdd(const Matrix A, const Matrix B, Matrix C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < C._width * C._height)
    {
        C._elements[i] = A._elements[i] + B._elements[i];
    }
}

__global__ void MatrixSubtract(const Matrix A, const Matrix B, Matrix C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < C._width * C._height)
    {
        C._elements[i] = A._elements[i] - B._elements[i];
    }
}

__global__ void MatrixMultiply(const Matrix A, const Matrix B,const Matrix C)
{
	 int Row = blockIdx.y * blockDim.y + threadIdx.y;
	 int Col = blockIdx.x * blockDim.x + threadIdx.x;

     __shared__ float As[TILE_SIZE][TILE_SIZE];
     __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0;

    int rowInBlock = threadIdx.y;
    int colInBlock = threadIdx.x;


    for (int i = 0; i < ((A._width + TILE_SIZE - 1) / TILE_SIZE); ++i)
    {
    	if((Row < A._height) && ((colInBlock + i*TILE_SIZE) < A._width))
    		As[rowInBlock][colInBlock] = A._elements[Row*A._width + colInBlock + i*TILE_SIZE];
    	else
    		As[rowInBlock][colInBlock] = 0;

    	if((Col < B._width) && ((rowInBlock + i*TILE_SIZE) < B._height))
    		Bs[rowInBlock][colInBlock] = B._elements[(rowInBlock + i*TILE_SIZE)*B._width + Col];
    	else
			Bs[rowInBlock][colInBlock] = 0;

        //Synchronize threads
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j)
        {
        		Cvalue += As[rowInBlock][j] * Bs[j][colInBlock];
        }

        __syncthreads();
    }

    if (Row < C._height && Col < C._width)//Saving Final result into Matrix C
    {
        C._elements[Row*C._width + Col] = Cvalue;
    }
}

