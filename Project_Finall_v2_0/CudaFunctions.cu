#include "CudaFunctions.h" // Header file
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "Matrix.h" // Header file of Matrix class


//Matrix addition kernel
//Pointers to array of elements of Matrix A, Matrix B and Matrix C. Values of Matrix A size.
__global__ void MatrixAdd(const float *A_elements, const float *B_elements,  float *C_elements, const int size)
{

	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;


	for(int i = thread; i < size; i += stride)
	 	//Modifying array of elements of Matrix C
		C_elements[i] = A_elements[i] + B_elements[i];
}

//Matrix subtraction kernel
//Pointers to array of elements of Matrix A, Matrix B and Matrix C. Values of Matrix A size.
__global__ void MatrixSubtract(const float* A_elements, const float* B_elements,  float* C_elements, const int size)
{
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;


	for(int i = thread; i < size; i += stride)
	 	//Modifying array of elements of Matrix C
		C_elements[i] = A_elements[i] - B_elements[i];
}
//Matrix multiplication kernel
//Pointers to array of elements of Matrix A, Matrix B and Matrix C.
//Values of Matrix A, Matrix B and Matrix C size.
__global__ void MatrixMultiply(const float* A_elements, const float* B_elements,  float* C_elements, const int X, const int Y, const int Z)
{
	int baseMatrixRow = blockIdx.y * blockDim.y + threadIdx.y;
	int baseMatrixCol = blockIdx.x * blockDim.x + threadIdx.x;

	int strideX = blockDim.x * gridDim.x;
	int strideY = blockDim.y * gridDim.y;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	for (int iterY = 0; iterY < (Y + strideY - 1) / strideY; iterY++)
	{
		for (int iterX = 0; iterX < (X + strideX - 1)/ strideX; iterX++)
		{
			int matrixRow = baseMatrixRow + strideY * (iterY);
			int matrixCol = baseMatrixCol + strideX * (iterX);

		    int blockRow = threadIdx.y;
		    int blockCol = threadIdx.x;

			float Cvalue = 0;

			for (int i = 0; i < ((X + TILE_SIZE - 1) / TILE_SIZE); ++i)
			{

		    	if((blockCol + i*TILE_SIZE) < X && matrixRow < Y)
		    		As[blockRow][blockCol] = A_elements[matrixRow * X + blockCol + i*TILE_SIZE];
		    	else
		    		As[blockRow][blockCol] = 0;

		    	if((blockRow + i*TILE_SIZE) < X && matrixCol < Z)
		    		Bs[blockRow][blockCol] = B_elements[(blockRow + i*TILE_SIZE) * Z + matrixCol];
		    	else
					Bs[blockRow][blockCol] = 0;

				//Synchronize threads
				__syncthreads();

				for (int j = 0; j < TILE_SIZE; ++j)
				{
					Cvalue += As[blockRow][j] * Bs[j][blockCol];
				}

				__syncthreads();
			}
		    if (matrixRow < Y && matrixCol < Z) //Saving Final result into Matrix C
		    {
		        C_elements[matrixRow * Z + matrixCol] = Cvalue;
		    }
		}
	}
}
//Matrix transposition kernel
//Pointers to array of elements of Matrix A and Matrix B. Values of Matrix A height and Matrix B width.
__global__ void MatrixTranspose(const float *A_elements, float *B_elements, const int A_width, const int A_height)
{
	int strideRow = blockDim.y * gridDim.y;
	int strideCol = blockDim.x * gridDim.x;

	for(int row = blockIdx.y * blockDim.y + threadIdx.y; row < A_width; row += strideRow)
		for(int col = blockIdx.x * blockDim.x + threadIdx.x; col < A_height; col += strideCol)
		{
			B_elements[row * A_height + col] = A_elements[col * A_width + row];
		}
}
//Matrix elements summation kernel
//Forwarded Matrix input and received pointer to output value.
//Calculate result using reduction.
__global__ void VectorSum(float *input, double *output, const int size)
{
	__shared__ double partialSum[2*BLOCK_SIZE];


	unsigned int thread = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
	//Each thread copy 2 elements from input into shared memory
	if (start + thread < size)
		partialSum[thread] = input[start+thread];
	else
		partialSum[thread] = 0;
	if (start + BLOCK_SIZE + thread < size)
		partialSum[BLOCK_SIZE + thread] = input[start + BLOCK_SIZE + thread];
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
