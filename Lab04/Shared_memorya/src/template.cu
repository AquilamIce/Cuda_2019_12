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


// defines
#define matrixType float // definition of matrixType variables
#define BLOCK_SIZE 16 // threads per block
#define SIZE 32 //size of square matrix


//Structure to store matrices
typedef struct {
    int width;
    int height;
    int stride;
    matrixType* elements;
} Matrix;




//Declarations of functions

// Single threaded matrix multiplication
void MatrixMul(const Matrix A, const Matrix B, Matrix C, int size);

// Show resoults of matrix multiplication
void MatrixPrint(Matrix C, int size);

// Get a element of matrix
__device__ float Get(const Matrix A, int row, int col);

// Set a element of matrix
__device__ void Set(Matrix A, int row, int col, matrixType value);

// Get a submatrix
__device__ Matrix GetSub(Matrix A, int row, int col);

//Definition of kernel to matrix multiplication
__global__ void MatrixMul(const Matrix, const Matrix, Matrix);


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
	Matrix A;
    A.stride = SIZE;
    A.width = SIZE;
    A.height = SIZE;
    Matrix B;
    B.stride = SIZE;
    B.width = SIZE;
    B.height = SIZE;
    Matrix C;
    C.stride = SIZE;
    C.width = SIZE;
    C.height = SIZE;
    Matrix C2;
	cudaMallocManaged(&A.elements, SIZE * SIZE * sizeof(matrixType));
	cudaMallocManaged(&B.elements, SIZE * SIZE * sizeof(matrixType));
	cudaMallocManaged(&C.elements, SIZE * SIZE * sizeof(matrixType));
    C2.elements = (matrixType*)calloc(SIZE, sizeof(matrixType));

    // Verify that allocations succeeded
    if (A.elements == NULL || B.elements == NULL || C.elements == NULL || C2.elements == NULL)
    {
        fprintf(stderr, "Failed to allocate matrices!\n");
        exit(EXIT_FAILURE);
    }

	//Initialize matrices
	for(int i = 0; i < SIZE * SIZE; i++)
	{
		A.elements[i] = rand()/(float)RAND_MAX;
		B.elements[i] = rand()/(float)RAND_MAX;
	}

	//Launch  Kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(SIZE / dimBlock.x, SIZE / dimBlock.y);
    MatrixMul<<<dimGrid, dimBlock>>>(A, B, C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//Check results on CPU
	MatrixMul(A, B, C2, SIZE);

    MatrixPrint(C, SIZE);
    MatrixPrint(C2, SIZE);

	for(int i = 0; i < SIZE * SIZE; ++i){

		if(fabs(C.elements[i] - C2.elements[i]) > 1e-5){

            fprintf(stderr, "Result verification failed at element %d!\t%f|%f\n", i, C.elements[i], C2.elements[i]);
            exit(EXIT_FAILURE);

		}

	}



    printf("Test PASSED\n");

    //Free memory !!!

	cudaFree(A.elements);
	cudaFree(B.elements);
	cudaFree(C.elements);
    free(C2.elements);

    printf("Done\n");
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Definition of signle threaded matrix multiplication function
void MatrixMul(const Matrix A, const Matrix B, Matrix C, int size)
{
	for(int row = 0; row < size; row ++)
	{
		for(int col = 0; col < size; col++)
		{
			for(int k = 0; k < size; k++)
				C.elements[(row*size) + col] += A.elements[(row*size) + k] * B.elements[col + (k*size)];
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Declaration of function to extract matrix
void MatrixPrint(Matrix C, int size){

	for(int row = 0; row < size; row ++){

		for(int col = 0; col < size; col++){

			printf("M[%d;%d] = %f\t", row + 1, col + 1, C.elements[(row*size) + col]);
		}

		printf("\n");
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void Set(Matrix A, int row, int col, matrixType value){

    A.elements[row * A.stride + col] = value;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ matrixType Get(const Matrix A, int row, int col){

    return A.elements[row * A.stride + col];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 __device__ Matrix GetSub(Matrix A, int row, int col) {

    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 __global__ void MatrixMul(Matrix A, Matrix B, Matrix C){


    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSub(C, blockRow, blockCol);

    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int i = 0; i < (SIZE / BLOCK_SIZE); ++i) {

        Matrix Asub = GetSub(A, blockRow, i);
        Matrix Bsub = GetSub(B, i, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = Get(Asub, row, col);
        Bs[row][col] = Get(Bsub, row, col);

        //Synchronize threads
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; ++j)
            Cvalue += As[row][j] * Bs[j][col];

        __syncthreads();
    }

    Set(Csub, row, col, Cvalue);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////








