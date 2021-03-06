#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "Functions.h"
#include "Matrix.h"

#define matrixType float
#define BLOCK_SIZE 16 //256 threads per block

void cuBLAS_multiply(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}

int main(int argc, char **argv)
{
	int widthA = 0;
	int heightA = 0;
	int widthB = 0;
	int heightB = 0;

	std::cout<<"Enter MatrixA width = MatrixB height"<<std::endl;
	std::cin >> widthA;
	heightB = widthA;
	std::cout<<"Enter MatrixA height = MatrixB width"<<std::endl;
	std::cin >> heightA;
	std::cout<<"Enter MatrixB width"<<std::endl;
	std::cin >> widthB;

	/*
	heightB = widthA = 4;
	heightA = 3;
	widthB = 4;
	*/
	Matrix matA = Matrix(widthA, heightA, cuda);
	Matrix matB = Matrix(widthB, heightB, cuda);
	InitializeMatrix(matA);
	InitializeMatrix(matB);

	int resWidth = matB._width;
	int resHeight = matA._height;

	Matrix matCNaive = Matrix(resWidth, resHeight, cuda);
	Matrix matCShared = Matrix(resWidth, resHeight, cuda);
	Matrix matCCublas = Matrix(resWidth, resHeight, cuda);

	Matrix matCCPU = Matrix(resWidth, resHeight);

	//Compute grid sizes
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((resWidth + dimBlock.x - 1)/dimBlock.x,(resHeight + dimBlock.x - 1)/dimBlock.y);

	//Launch Kernels
	double ti = cpuTimer();
	MatrixMulShared<<<dimGrid, dimBlock>>>(matA, matB, matCShared);
	cudaDeviceSynchronize();
	double sharedTime = cpuTimer() - ti;
	ti = cpuTimer();
	MatrixMulNaive<<<dimGrid, dimBlock>>>(matA , matB, matCNaive);
	cudaDeviceSynchronize();
	double naiveTime = cpuTimer() - ti;
	//Multiply using cuBLAS
	ti = cpuTimer();
	//

	CublasMultiply(matA, matB, matCCublas);


	double cublasTime = cpuTimer() - ti;
	//Check results on CPU
	MatrixMul(matA, matB, matCCPU);


	std::cout<<"Results:"<<std::endl
			<<"Naive: "<<naiveTime<<std::endl
			<<"Shared: "<<sharedTime<<std::endl
			<<"cuBLAS: "<<cublasTime<<std::endl<<std::endl;
	std::cout<<"Checking shared memory version"<<std::endl;
	MatrixCompare(matCCPU, matCShared);
	std::cout<<"Checking naive version"<<std::endl;
	MatrixCompare(matCCPU, matCNaive);
	std::cout<<"Checking cuBLAS version"<<std::endl;
	MatrixCompare(matCCPU, matCCublas);

}
