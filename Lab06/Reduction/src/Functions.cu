#include "Functions.h"


void InitializeVector(VectorType *input, int size)
{
	for (int i = 0; i < size; i++)
	{
		input[i] = rand()/(VectorType)RAND_MAX;
	}
}

VectorType TotalCPU(VectorType *input, int size)
{
	VectorType output = 0;
	for (int i = 0; i < size; i++)
	{
		output += input[i];
	}
	return output;
}


__global__ void ReductionTotalKernel(VectorType *input, VectorType *output, int size)
{
	__shared__ VectorType partialSum[2*BLOCK_SIZE];

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
	for (int stride =BLOCK_SIZE; stride >= 1; stride >>= 1)
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

__global__ void NaiveTotalKernel(VectorType *input, VectorType *output, int size)
{
	if(BLOCK_SIZE * blockIdx.x + threadIdx.x  < size)
		atomicAdd(output, input[BLOCK_SIZE * blockIdx.x + threadIdx.x]);
}


