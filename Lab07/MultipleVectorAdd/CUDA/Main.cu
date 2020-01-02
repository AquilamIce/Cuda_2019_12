#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
//#include <cudaProfiler.h>
#include <cstdio>
#include <sys/time.h>


//#define SIZE (1024)
#define BLOCK_SIZE 16

__global__ void VectorAdd(float *VecA, float *VecB, float *VecC, int size);
void InitializeVector(float *VecA, int size);

int main(int argc, char *argv[])
{

	int SIZE = atoi(argv[1]);

	struct timeval start, end;
	gettimeofday(&start, NULL);

	//Input
	float *VecA, *VecB, *VecE, *VecG;
	//Output
	float *VecC, *VecD,  *VecF;
	cudaMallocManaged(&VecA, SIZE * sizeof(float));
	cudaMallocManaged(&VecB, SIZE * sizeof(float));
	cudaMallocManaged(&VecC, SIZE * sizeof(float));
	cudaMallocManaged(&VecD, SIZE * sizeof(float));
	cudaMallocManaged(&VecE, SIZE * sizeof(float));
	cudaMallocManaged(&VecF, SIZE * sizeof(float));
	cudaMallocManaged(&VecG, SIZE * sizeof(float));

	//Initialize input
	InitializeVector(VecA, SIZE);
	InitializeVector(VecB, SIZE);
	InitializeVector(VecE, SIZE);
	InitializeVector(VecG, SIZE);

	//Calculate grid dimensions
	dim3 dimBlock = BLOCK_SIZE;
	dim3 dimGrid((SIZE + BLOCK_SIZE)/(BLOCK_SIZE));

	//Launch kernels
	VectorAdd<<<dimGrid, dimBlock>>>(VecA, VecB, VecC, SIZE);
	cudaDeviceSynchronize();
	VectorAdd<<<dimGrid, dimBlock>>>(VecC, VecE, VecD, SIZE);
	cudaDeviceSynchronize();
	VectorAdd<<<dimGrid, dimBlock>>>(VecD, VecG, VecF, SIZE);
	cudaDeviceSynchronize();

	gettimeofday(&end, NULL);
	double diff = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

	std::cout<<"Timing [us]: "<<diff<<std::endl;


	//Check results
    for (int i = 0; i < SIZE; i++)
    {
        if (VecA[i] + VecB[i] + VecE[i] + VecG[i] - VecF[i]  > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test passed\n");

    //Free memory
    cudaFree(VecA);
    cudaFree(VecB);
    cudaFree(VecC);
    cudaFree(VecD);
    cudaFree(VecE);
    cudaFree(VecF);
    cudaFree(VecG);

}

__global__ void VectorAdd(float *VecA, float *VecB, float *VecC, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
		VecC[i] = VecA[i] + VecB[i];
}
void InitializeVector(float *VecA, int size)
{
	for (int i = 0; i < size; i++)
		VecA[i] = rand()/(float)RAND_MAX;
}
