#include "MatrixCalculator.h"
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <cuda_runtime.h>
#include "Matrix.h"
#include "CudaFunctions.h"
#include "Functions.h"
MatrixCalculator::MatrixCalculator()
{
	cudaGetDevice(&deviceId);


	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceId);

	SMCount = props.multiProcessorCount;
}

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
	int width = 2000;
	int height = 2000;

	std::cout<<"Set size of matrix: "<<std::endl;

	std::cout<<"Width: ";
	//GetInt(&width);
	std::cout<<std::endl;

	std::cout<<"Height: ";
	//GetInt(&height);
	std::cout<<std::endl;

	std::cout<<"Matrix size: "<<width<<" x "<<height<<std::endl;
	//Create input matrices filled with random values
	Matrix A(width, height);
	Matrix B(width, height);

    cudaMemPrefetchAsync(A._elements, A._width * A._height * sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(B._elements, B._width * B._height * sizeof(float), cudaCpuDeviceId);

	A.Initialize();
	B.Initialize();

	//Create matrices for results
	Matrix C_CPU = Matrix(width, height);
	Matrix C_GPU = Matrix(width, height);

	cudaMemPrefetchAsync(C_CPU._elements, C_CPU._width * C_CPU._height * sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(C_GPU._elements, C_GPU._width * C_GPU._height * sizeof(float), deviceId);

	//Define grid configuration
	dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
	dim3 dimGrid(128*SMCount);

	//Compute operation on CPU
    std::cout<<std::endl;
    clock_t start = clock();
    C_CPU = A + B;
    std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;

    cudaMemPrefetchAsync(A._elements, A._width * A._height * sizeof(float), deviceId);
	cudaMemPrefetchAsync(B._elements, B._width * B._height * sizeof(float), deviceId);

    //Compute on GPU
    start = clock();
    MatrixAdd<<<dimGrid, dimBlock>>>(A._elements, B._elements, C_GPU._elements, A._width * A._height);
	cudaDeviceSynchronize();
    std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;

	cudaMemPrefetchAsync(C_GPU._elements, C_GPU._width * C_GPU._height * sizeof(float), cudaCpuDeviceId);

    std::cout<<"Validation: "<<std::endl;
    Matrix::MatrixCompare(C_CPU,C_GPU);
}

void MatrixCalculator::MatrixSubtraction()
{
	std::cout<<"Subtraction"<<std::endl;

	//Matrix size
	int width = 2000;
	int height = 2000;

	std::cout<<"Set size of matrix: "<<std::endl;

	std::cout<<"Width: ";
	//GetInt(&width);
	std::cout<<std::endl;

	std::cout<<"Height: ";
	//GetInt(&height);
	std::cout<<std::endl;

	std::cout<<"Matrix size: "<<width<<" x "<<height<<std::endl;
	//Create input matrices filled with random values
	Matrix A(width, height);
	Matrix B(width, height);

    cudaMemPrefetchAsync(A._elements, A._width * A._height * sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(B._elements, B._width * B._height * sizeof(float), cudaCpuDeviceId);

	A.Initialize();
	B.Initialize();

	//Create matrices for results
	Matrix C_CPU = Matrix(width, height);
	Matrix C_GPU = Matrix(width, height);

	cudaMemPrefetchAsync(C_CPU._elements, C_CPU._width * C_CPU._height * sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(C_GPU._elements, C_GPU._width * C_GPU._height * sizeof(float), deviceId);

	//Define grid configuration
	dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
	dim3 dimGrid(128*SMCount);

	//Compute operation on CPU
    std::cout<<std::endl;
    clock_t start = clock();
    C_CPU = A - B;
    std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;

    cudaMemPrefetchAsync(A._elements, A._width * A._height * sizeof(float), deviceId);
	cudaMemPrefetchAsync(B._elements, B._width * B._height * sizeof(float), deviceId);

    //Compute on GPU
    start = clock();
    MatrixSubtract<<<dimGrid, dimBlock>>>(A._elements, B._elements, C_GPU._elements, A._width * A._height);
	cudaDeviceSynchronize();
    std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;

	cudaMemPrefetchAsync(C_GPU._elements, C_GPU._width * C_GPU._height * sizeof(float), cudaCpuDeviceId);

    std::cout<<"Validation: "<<std::endl;
    Matrix::MatrixCompare(C_CPU,C_GPU);
}

void MatrixCalculator::MatrixMultiplication()
{
	std::cout<<"Multiplication"<<std::endl;

	//Matrix size
	int X = 111, Y = 222, Z = 333;

	std::cout<<"Set size of matrix: "<<std::endl;
	std::cout<<"width A = height B: "<<std::endl;
	//GetInt(&X);
	std::cout<<"height A: "<<std::endl;
	//GetInt(&Y);
	std::cout<<"width B: "<<std::endl;
	//GetInt(&Z);
	std::cout<<std::endl;

	//Create input matrices filled with random values
	Matrix A(X, Y);
	Matrix B(Z, X);

    cudaMemPrefetchAsync(A._elements, A._width * A._height * sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(B._elements, B._width * B._height * sizeof(float), cudaCpuDeviceId);

	A.Initialize();
	B.Initialize();

	//Create matrices for results
	Matrix C_CPU = Matrix(Z, Y);
	Matrix C_GPU = Matrix(Z, Y);

	cudaMemPrefetchAsync(C_CPU._elements, C_CPU._width * C_CPU._height * sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(C_GPU._elements, C_GPU._width * C_GPU._height * sizeof(float), deviceId);

	//Define grid configuration
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(20 * SMCount, 20 * SMCount);

	//Compute on CPU
    clock_t start = clock();
    C_CPU = A * B;
    std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;

    cudaMemPrefetchAsync(A._elements, A._width * A._height * sizeof(float), deviceId);
	cudaMemPrefetchAsync(B._elements, B._width * B._height * sizeof(float), deviceId);

    //Compute on GPU
    std::cout<<std::endl;
    start = clock();
    MatrixMultiply<<<dimGrid, dimBlock>>>(A._elements, B._elements, C_GPU._elements, A._width, A._height, B._width);
    cudaDeviceSynchronize();
    std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
    std::cout<<std::endl;

	cudaMemPrefetchAsync(C_GPU._elements, C_GPU._width * C_GPU._height * sizeof(float), cudaCpuDeviceId);

    //Check Results
    std::cout<<"Validation: "<<std::endl;
    Matrix::MatrixCompare(C_CPU,C_GPU);
}

void MatrixCalculator::MatrixTransposition()
{
	std::cout<<"Transposition"<<std::endl;
	//Matrix size
	int width = 1000, height = 2000;

	std::cout<<"Set size of matrix: "<<std::endl;

	std::cout<<"Width: ";
	//GetInt(&width);
	std::cout<<std::endl;

	std::cout<<"Height: ";
	//GetInt(&height);
	std::cout<<std::endl;

	std::cout<<"Matrix size: "<<width<<" x "<<height<<std::endl;
	//Create input matrices filled with random values

	//Create input matrix filled with random values
	Matrix A(width, height);

    cudaMemPrefetchAsync(A._elements, A._width * A._height * sizeof(float), cudaCpuDeviceId);

	A.Initialize();

	//Create matrices for results
	Matrix C_CPU(height, width);
	Matrix C_GPU(height, width);

	cudaMemPrefetchAsync(C_CPU._elements, C_CPU._width * C_CPU._height * sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(C_GPU._elements, C_GPU._width * C_GPU._height * sizeof(float), deviceId);

	//Define grid configuration
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(32 * SMCount, 32 * SMCount);

	//Compute on CPU
	clock_t start = clock();
	C_CPU = A.Transpose();
	std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
	std::cout<<std::endl;

	//Compute on GPU
	std::cout<<std::endl;
	start = clock();
	MatrixTranspose<<<dimGrid, dimBlock>>>(A._elements, C_GPU._elements, A._width,A._height);
	cudaDeviceSynchronize();
	std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
	std::cout<<std::endl;

	cudaMemPrefetchAsync(C_GPU._elements, C_GPU._width * C_GPU._height * sizeof(float), cudaCpuDeviceId);

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
	Matrix A(width, 1);
	
	cudaMemPrefetchAsync(A._elements, A._width * A._height * sizeof(float), cudaCpuDeviceId);

	A.Initialize();
	
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
	
	cudaMemPrefetchAsync(A._elements, A._width * A._height * sizeof(float), deviceId);

	//Compute on GPU
	start = clock();
	VectorSum<<<dimGrid, dimBlock>>>(A._elements, C_GPU, A._width);
	cudaDeviceSynchronize();
	std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
	std::cout<<std::endl;

	std::cout<<"Validation: "<<std::endl;
	if(fabs(C_CPU - *C_GPU) > 1e-3)
	{
		fprintf(stderr, "Verification failed %f|%f\n", C_CPU, *C_GPU);
	}
}
