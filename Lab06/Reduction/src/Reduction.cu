// includes, system
#include <stdlib.h>
#include <iostream>
#include <cmath>
// includes CUDA
#include <cuda_runtime.h>
//#include <time.h>
#include <sys/time.h>

#include "Functions.h"

//#define SIZE 100000



int main(int argc, char *argv[])
{


	int SIZE = atoi(argv[1]);
	//Allocate memory
	VectorType *input;
	cudaMallocManaged(&input, SIZE * sizeof(VectorType));

	VectorType *outputReduction;
	cudaMallocManaged(&outputReduction, sizeof(VectorType));

	VectorType *outputNaive;
	cudaMallocManaged(&outputNaive, sizeof(VectorType));

	InitializeVector(input, SIZE);

	//Calculate grid dimensions
	dim3 dimBlock = BLOCK_SIZE;
	//Each block takes 2 * BLOCK_SIZE elements
	dim3 dimGrid =(SIZE + 2 * BLOCK_SIZE - 2)/(2 * BLOCK_SIZE);

	//Calculate total using reduction algorithm
	ReductionTotalKernel<<<dimGrid, dimBlock>>>(input, outputReduction, SIZE);
	cudaDeviceSynchronize();

	//Change number of blocks
	dimGrid = (SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE;

	//Calculate total using naive algorithm
	NaiveTotalKernel<<<dimGrid, dimBlock>>>(input, outputNaive, SIZE);
	cudaDeviceSynchronize();

	struct timeval start, end;
	gettimeofday(&start, NULL);


	//Calculate total using CPU
	VectorType totalCPU{TotalCPU(input, SIZE)};

	gettimeofday(&end, NULL);
	double diff = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));


	std::cout<<"CPU Result: "<<totalCPU<<std::endl;
	std::cout<<"CPU Timing [us]: "<<diff<<std::endl;
	std::cout<<"GPUReduction Result: "<<*outputReduction<<std::endl;
	std::cout<<"GPUNaive Result: "<<*outputNaive<<std::endl;

	//Check results
	VectorType eps = 0.01;
	if (abs(totalCPU - *outputReduction) < eps && abs(totalCPU - *outputNaive) < eps)
	{
		std::cout<<"Test passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test failed"<<std::endl;
	}

	cudaFree(input);
	cudaFree(outputNaive);
	cudaFree(outputReduction);
}
