////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>       /* time */


// includes CUDA
#include <cuda_runtime.h>

// includes, project
//#include <helper_cuda.h>
//#include <helper_functions.h> // helper functions for SDK examples

#define matrixType float // definition of matrixType variables
#define BLOCK_SIZE 16 //256 threads per block
#define SIZE 16 //size of square matrix


//functions' declarations & definitions
void MatrixMul(float *A, float *B, float *C, int size);
void MatrixPrint(float *C, int size);
void MatrixMul(float *, float *, float *, int);


//Definition of naive kernel to matrix multiplication
__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if((Row < Width) && (Col < Width))
	{
		float Pvalue = 0;
		for(int k = 0; k < Width; ++k)
		{
			Pvalue += M[Row*Width+k]*N[k*Width+Col];
		}
		P[Row*Width+Col] = Pvalue;
	}
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

	 /* initialize random seed: */
	 srand (time(NULL));

	int deviceCount = 0;
	//Check available memory
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    if(deviceProp.totalGlobalMem < SIZE * SIZE * sizeof(matrixType))
    {
    	fprintf(stderr, "Data size too large\n");
        exit(EXIT_FAILURE);
    }


	//Allocate memory
	matrixType *matA;
	matrixType *matB;
	matrixType *matC;
	matrixType *matC2 = (matrixType*)calloc(SIZE, sizeof(matrixType));
	cudaMallocManaged(&matA, SIZE * SIZE * sizeof(matrixType));
	cudaMallocManaged(&matB, SIZE * SIZE * sizeof(matrixType));
	cudaMallocManaged(&matC, SIZE * SIZE * sizeof(matrixType));

    // Verify that allocations succeeded
    if (matA == NULL || matB == NULL || matC == NULL || matC2 == NULL)
    {
        fprintf(stderr, "Failed to allocate matrices!\n");
        exit(EXIT_FAILURE);
    }

	//Initialize matrices
	for(int i = 0; i < SIZE * SIZE; i++)
	{
		matA[i] = rand()/(float)RAND_MAX;
		matB[i] = rand()/(float)RAND_MAX;
	}

	//Launch naive Kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((SIZE + dimBlock.x - 1)/dimBlock.x,(SIZE + dimBlock.x - 1)/dimBlock.y);
	MatrixMulKernel<<<dimGrid, dimBlock>>>(matA, matB, matC, SIZE);
	cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch Matrixmultiplication kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//Check results on CPU
	MatrixMul(matA, matB, matC2, SIZE);

	MatrixPrint(matC, SIZE);

	MatrixPrint(matC2, SIZE);

	for(int i = 0; i < SIZE * SIZE; ++i){

		if(fabs(matC[i] - matC2[i]) > 1e-4){

            fprintf(stderr, "Result verification failed at element %d!\t%f|%f\n", i, matC[i], matC2[i]);
            exit(EXIT_FAILURE);

		}

	}



    printf("Test PASSED\n");

	cudaFree(matA);
	cudaFree(matB);
	cudaFree(matC);
	free(matC2);
    printf("Done\n");
    return 0;
}

//Definition of signle threaded matrix multiplication function
void MatrixMul(float *A, float *B, float *C, int size)
{
	for(int row = 0; row < size; row ++)
	{
		for(int col = 0; col < size; col++)
		{
			for(int k = 0; k < size; k++)
				C[(row*size) + col] += A[(row*size) + k] * B[col + (k*size)];
		}
	}
}
//Declaration of function to extract matrix
void MatrixPrint(float *C, int size)
{
	for(int row = 0; row < size; row ++)
	{
		for(int col = 0; col < size; col++)
		{
			printf("M[%d;%d] = %f\t", row + 1, col + 1, C[(row*size) + col]);
		}
		printf("\n");
	}
}
