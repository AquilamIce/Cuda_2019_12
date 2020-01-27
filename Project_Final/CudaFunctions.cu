#include "CudaFunctions.h" // Header file
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "Matrix.h" // Header file of Matrix class


//Matrix addition kernel
//Pointers to array of elements of Matrix A, Matrix B and Matrix C. Values of Matrix A size.
__global__ void MatrixAdd(const float *A_elements, const float *B_elements,  float *C_elements, const int A_width, const int A_height)
{

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < A_height && col < A_width)
	{
	 	//Modifying array of elements of Matrix C
		C_elements[row * A_width + col] = A_elements[row * A_width + col] + B_elements[row * A_width + col];
	}
}

//Matrix substraction kernel
//Pointers to array of elements of Matrix A, Matrix B and Matrix C. Values of Matrix A size.
__global__ void MatrixSubtract(const float* A_elements, const float* B_elements,  float* C_elements, const int A_width, const int A_height)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < A_height && col < A_width)
	{
	 	//Modifying array of elements of Matrix C
		C_elements[row * A_width + col] = A_elements[row * A_width + col] - B_elements[row * A_width + col];
	}
}

//Matrix multiplication kernel
//Pointers to array ofelements of Matrix A, Matrix B and Matrix C.
//Values of Matrix A, Matrix B and Matrix C size.
__global__ void MatrixMultiply(const float* A_elements, const float* B_elements,  float* C_elements, const int A_width, const int A_height, const int B_width, const int B_height, int C_width,  int C_height)
{
	 int Row = blockIdx.y * blockDim.y + threadIdx.y;
	 int Col = blockIdx.x * blockDim.x + threadIdx.x;

     __shared__ float As[TILE_SIZE][TILE_SIZE];
     __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0;

    int rowInBlock = threadIdx.y;
    int colInBlock = threadIdx.x;


    for (int i = 0; i < ((A_width + TILE_SIZE - 1) / TILE_SIZE); ++i)
    {
    	if((Row < A_height) && ((colInBlock + i*TILE_SIZE) < A_width))
    		As[rowInBlock][colInBlock] = A_elements[Row*A_width + colInBlock + i*TILE_SIZE];
    	else
    		As[rowInBlock][colInBlock] = 0;

    	if((Col < B_width) && ((rowInBlock + i*TILE_SIZE) < B_height))
    		Bs[rowInBlock][colInBlock] = B_elements[(rowInBlock + i*TILE_SIZE)*B_width + Col];
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

    if (Row < C_height && Col < C_width) //Saving Final result into Matrix C
    {
        C_elements[Row*C_width + Col] = Cvalue;
    }
}

//Matrix transposition kernel
//Pointers to array of elements of Matrix A and Matrix B. Values of Matrix A height and Matrix B width.
__global__ void MatrixTranspose(const float *A_elements, float *B_elements, const int A_width, const int A_height, const int B_width)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < A_width && col < A_height)
	{
		B_elements[row * B_width + col] = A_elements[col * A_width + row];
	}
}

//Matrix elements summation kernel
//Forwarded Matrix input and recived pointer to output value.
__global__ void VectorSum(Matrix input, double *output)
{
	__shared__ double partialSum[2*BLOCK_SIZE];

	unsigned int thread = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
	//Each thread copy 2 elements from input into shared memory
	if (start + thread < input._width)
		partialSum[thread] = input._elements[start+thread];
	else
		partialSum[thread] = 0;
	if (start + BLOCK_SIZE + thread < input._width)
		partialSum[BLOCK_SIZE + thread] = input._elements[start + BLOCK_SIZE + thread];
	else
		partialSum[BLOCK_SIZE + thread] = 0;

	//Use reduction to calculate sum of each block
	for (int stride{BLOCK_SIZE}; stride >= 1; stride >>= 1)
	{
		__syncthreads();
		if (thread < stride)
			partialSum[thread] += partialSum[thread + stride];
	}

	//Save sum of each block into output
	if (thread == 0)
	{
		//Atomic function guarantees that threads will not try to write into memory at the same time
		atomicAdd(output, partialSum[0]);
	}
}
