#include "Functions.h"

#include "cublas_v2.h"
#include <cuda_runtime.h>

double cpuTimer()
{
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return((double)clock.tv_sec + (double)clock.tv_usec * 1.e-6);
}

void MatrixMul(Matrix &A, Matrix &B, Matrix &C)
{
	for(int row = 0; row < C._height; row ++)
	{
		for(int col = 0; col < C._width; col++)
		{
			for(int k = 0; k < A._width; k++)
				C._elements[(row*C._width) + col] += A._elements[(row*A._width) + k] * B._elements[col + (k*B._width)];
		}
	}
}
void MatrixPrint(Matrix &A)
{
	for(int row = 0; row < A._height; row ++)
	{
		for(int col = 0; col < A._width; col++)
		{
			printf("M[%d;%d] = %f\t", row + 1, col + 1, A._elements[(row*A._width) + col]);
		}
		printf("\n");
	}
}
void MatrixCompare(Matrix &A, Matrix &B)
{
	for(int row = 0; row < A._height; row++)
	{	for(int col = 0; col < A._width; col++)
		{
			if(fabs(A._elements[(row*A._width) + col] - B._elements[(row*B._width) + col]) > 1e-4)
			{
				fprintf(stderr, "Result verification failed at element M[%d;%d]!\t%f|%f\n", row+1, col+1, A._elements[(row*A._width) + col], B._elements[(row*B._width) + col]);
				exit(EXIT_FAILURE);
			}
		}
	}
	std::cout<<"Check ok"<<std::endl;
}
void InitializeMatrix(Matrix &A, int val)
{
	for(int i = 0; i < A._width * A._height; i++)
		A._elements[i] = val;
}
void InitializeMatrix(Matrix &A)
{
	for(int i = 0; i < A._width * A._height; i++)
		A._elements[i] = rand()/(float)RAND_MAX;
}

void CublasMultiply(Matrix &matA, Matrix &matB, Matrix &matC)
{
	int m = matB._width;
	int n = matA._height;
	int k = matA._width;

	int lda = matB._width;
	int ldb = matA._width;
	int ldc = matB._width;

	float *A = matA._elements;
	float *B = matB._elements;
	float *C = matC._elements;

	const float a = 1;
	const float b = 0;
	const float *alpha = &a;
	const float *beta = &b;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, lda, A, ldb, beta, C, ldc);


	cublasDestroy(handle);
}

__global__ void MatrixMulNaive(Matrix A, Matrix B, Matrix C)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if((Row < C._height) && (Col < C._width))
	{
		float Pvalue = 0;
		for(int k = 0; k < A._width; ++k)
		{
			Pvalue += A._elements[Row*A._width+k]*B._elements[k*B._width+Col];
		}
		C._elements[Row*C._width+Col] = Pvalue;
	}
}

__global__ void MatrixMulShared(Matrix A, Matrix B, Matrix C)
{


	 int Row = blockIdx.y * blockDim.y + threadIdx.y;
	 int Col = blockIdx.x * blockDim.x + threadIdx.x;

     __shared__ float As[TILE_SIZE][TILE_SIZE];
     __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int i = 0; i < ((A._width+TILE_SIZE-1)/ TILE_SIZE); ++i) {


    	if((Row < A._height) && ((col + i*TILE_SIZE) < A._width))
    		As[row][col] = A._elements[Row*A._width + col + i*TILE_SIZE];
    	else
    		As[row][col] = 0;
    	if((Col < B._width) && ((row + i*TILE_SIZE) < B._height))
    		Bs[row][col] = B._elements[(row + i*TILE_SIZE)*B._width + Col];
    	else
			Bs[row][col] = 0;
        //Synchronize threads
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j)
        {
        		Cvalue += As[row][j] * Bs[j][col];
        }

        __syncthreads();
    }

    if (Row < C._height && Col < C._width)//Saving Final result into Matrix C
    {
        C._elements[Row*C._width + Col] = Cvalue;
    }
}
