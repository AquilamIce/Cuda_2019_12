#include "MatrixCalculator.h"

#include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include <cuda_runtime.h>

#include "Matrix.h"
#include "CudaFunctions.h"

Matrix MatrixCalculator::CreateRandomMatrix(int width, int height)
{
	Matrix A = Matrix(width, height);
	A.Initialize(0, true);
	return A;
}


void MatrixCalculator::MatrixAddition()
{
	std::cout<<"Addition"<<std::endl;

	//Matrix size
	int width, height;

	std::cout<<"Set size of matrix: width, height..."<<std::endl;
	std::cin>>width;
	std::cin>>height;
	std::cout<<std::endl;

	//Create input matrices filled with random values
	Matrix A = CreateRandomMatrix(width, height);
	Matrix B = CreateRandomMatrix(width, height);

	//Create matrices for results
	Matrix C_CPU = Matrix(width, height);
	Matrix C_GPU = Matrix(width, height);

	//Define grid configuration
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((width + dimBlock.x - 1)/dimBlock.x, (height + dimBlock.y - 1)/dimBlock.y);

	//Compute operation on CPU
    std::cout<<std::endl;
    clock_t start = clock();
    C_CPU = A + B;
    std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;

    //Compute on GPU
    start = clock();
    MatrixAdd<<<dimGrid, dimBlock>>>(A._elements, B._elements, C_GPU._elements, A._width, A._height);
	cudaDeviceSynchronize();
    std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;


    std::cout<<"Validation: "<<std::endl;
    Matrix::MatrixCompare(C_CPU,C_GPU);
}

void MatrixCalculator::MatrixSubtraction()
{
	std::cout<<"Subtraction"<<std::endl;

	//Matrix size
	int width, height;

	std::cout<<"Set size of matrix: width, height..."<<std::endl;
	std::cin>>width;
	std::cin>>height;
	std::cout<<std::endl;

	//Create input matrices filled with random values
	Matrix A = CreateRandomMatrix(width, height);
	Matrix B = CreateRandomMatrix(width, height);

	//Create matrices for results
	Matrix C_CPU = Matrix(width, height);
	Matrix C_GPU = Matrix(width, height);

	//Define grid configuration
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((width + dimBlock.x - 1)/dimBlock.x, (height + dimBlock.y - 1)/dimBlock.y);

	//Compute operation on CPU
    std::cout<<std::endl;
    clock_t start = clock();
    C_CPU = A - B;
    std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;

    //Compute on GPU
    start = clock();
    MatrixSubtract<<<dimGrid, dimBlock>>>(A._elements, B._elements, C_GPU._elements, A._width, A._height);
	cudaDeviceSynchronize();
    std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Validation: "<<std::endl;
    Matrix::MatrixCompare(C_CPU,C_GPU);
}

void MatrixCalculator::MatrixMultiplication()
{
	std::cout<<"Multiplication"<<std::endl;

	//Matrix size
	int X, Y ,Z;

	std::cout<<"Set size of matrix: width A = height B, height A, width B..."<<std::endl;
	std::cin>>X;
	std::cin>>Y;
	std::cin>>Z;

	std::cout<<std::endl;

	//Create input matrices filled with random values
	Matrix A = CreateRandomMatrix(X, Y);
	Matrix B = CreateRandomMatrix(Z, X);

	//Create matrices for results
	Matrix C_CPU = Matrix(Z, Y);
	Matrix C_GPU = Matrix(Z, Y);

	//Define grid configuration
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((C_GPU._width + dimBlock.x - 1)/dimBlock.x, (C_GPU._height + dimBlock.y - 1)/dimBlock.y);

	//Compute on CPU
    clock_t start = clock();
    C_CPU = A * B;
    std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;

    //Compute on GPU
    std::cout<<std::endl;
    start = clock();
    MatrixMultiply<<<dimGrid, dimBlock>>>(A._elements, B._elements, C_GPU._elements, A._width, A._height, B._height, B._width, C_GPU._height, C_GPU._width);
    cudaDeviceSynchronize();
    std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;

    //Check Results
    std::cout<<"Validation: "<<std::endl;
    Matrix::MatrixCompare(C_CPU,C_GPU);
}

void MatrixCalculator::MatrixTransposition()
{
	std::cout<<"Transposition"<<std::endl;
	//Matrix size
	int width, height;

	std::cout<<"Set size of matrix: width, height..."<<std::endl;
	std::cin>>width;
	std::cin>>height;
	std::cout<<std::endl;

	//Create input matrix filled with random values
	Matrix A = CreateRandomMatrix(width, height);

	//Create matrices for results
	Matrix C_CPU = Matrix(height, width);
	Matrix C_GPU = Matrix(height, width);

	//Define grid configuration
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((C_GPU._width + dimBlock.x - 1)/dimBlock.x, (C_GPU._height + dimBlock.y - 1)/dimBlock.y);

	//Compute on CPU
	clock_t start = clock();
	C_CPU = A.Transpose();
	std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
	std::cout<<std::endl;

	//Compute on GPU
	std::cout<<std::endl;
	start = clock();
	MatrixTranspose<<<dimGrid, dimBlock>>>(A._elements, C_GPU._elements,A._width,A._height, C_GPU._width);
	cudaDeviceSynchronize();
	std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
	std::cout<<std::endl;

	//Check Results
	std::cout<<"Validation: "<<std::endl;
	Matrix::MatrixCompare(C_CPU,C_GPU);
}

void MatrixCalculator::VectorSummation()
{
	std::cout<<"Vector Sum"<<std::endl;

	//Matrix size
	int width;

	std::cout<<"Set length of vector..."<<std::endl;
	std::cin>>width;
	std::cout<<std::endl;

	//Create input matrices filled with random values
	Matrix A = CreateRandomMatrix(width, 1);

	//Create matrices for results
	double C_CPU = 0;

	double *C_GPU;
	cudaMallocManaged(&C_GPU, sizeof(double));

	//Define grid configuration
	dim3 dimBlock(BLOCK_SIZE);
	//Each block takes 2 * BLOCK_SIZE elements
	dim3 dimGrid(((unsigned)width + 2 * BLOCK_SIZE - 2)/(2 * BLOCK_SIZE));

	//Compute operation on CPU
	std::cout<<std::endl;
	clock_t start = clock();
	C_CPU = A.VectorSum();
	std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
	std::cout<<std::endl;
	std::cout<<std::endl;

	//Compute on GPU
	start = clock();
	VectorSum<<<dimGrid, dimBlock>>>(A, C_GPU);
	cudaDeviceSynchronize();
	std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
	std::cout<<std::endl;

	std::cout<<"Validation: "<<std::endl;
	if(fabs(C_CPU - *C_GPU) > 1e-3)
	{
		fprintf(stderr, "Verification failed %f|%f\n", C_CPU, *C_GPU);
	}
}
