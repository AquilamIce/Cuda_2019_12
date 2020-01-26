#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "Matrix.h"
#include "CudaFunctions.h"

#define BLOCK_SIZE 16 //256 threads per block

int main(int argc, char **argv)
{
	Matrix A = Matrix(40, 40);
	Matrix B = Matrix(40, 40);

	A.Initialize(3);
	B.Initialize(2);

	Matrix F = A * B;
	Matrix G = Matrix(40, 40);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((G._width + dimBlock.x - 1)/dimBlock.x, (G._height + dimBlock.y - 1)/dimBlock.y);

	MatrixMultiply<<<dimGrid, dimBlock>>>(A, B, G);

	cudaDeviceSynchronize();

	G.Print();

	F.Print();
}
