#include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include <cuda_runtime.h>
#include "Matrix.h"
#include "CudaFunctions.h"

#define BLOCK_SIZE 16 //256 threads per block

int main(int argc, char **argv)
{



int width = 0;
int height = 0;
float first = 0;
float second = 0;
char b_print;
int o_print;

std::cout<<"DEMO OF COMPUTING PROGRAM"<<std::endl;
std::cout<<"Set size of matrix: width, height..."<<std::endl;
std::cin>>width;
std::cin>>height;
std::cout<<std::endl;


std::cout<<"Matrix A"<<std::endl;
std::cout<<"Fill in with a number..."<<std::endl;
std::cin>>first;
std::cout<<std::endl;

std::cout<<"Matrix B"<<std::endl;
std::cout<<"Fill in with a number..."<<std::endl;
std::cin>>second;
std::cout<<std::endl;

Matrix A = Matrix(width,height);
Matrix B = Matrix(width,height);
Matrix C_GPU = Matrix(width,height);

A.Initialize(first);
B.Initialize(second);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((C_GPU._width + dimBlock.x - 1)/dimBlock.x, (C_GPU._height + dimBlock.y - 1)/dimBlock.y);

std::cout<<"Print matrices (Y/N):"<<std::endl;
std::cin>>b_print;



if(b_print=='Y')
{
    std::cout<<"Matrix A"<<std::endl;
    A.Print();
    std::cout<<std::endl;

    std::cout<<"Matrix B"<<std::endl;
    B.Print();
    std::cout<<std::endl;
}

std::cout<<"Choose operation: Add - 0, Sub - 1, Mult - 2"<<std::endl;
std::cin>>o_print;
std::cout<<std::endl;




switch(o_print)
{
    case 0:
        {
        std::cout<<"Matrix addition: A+B"<<std::endl;
        std::cout<<std::endl;
        clock_t start = clock();
        Matrix C_CPU = A + B;
        std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
        std::cout<<std::endl;
        if(b_print=='Y') C_CPU.Print();


        std::cout<<std::endl;
        start = clock();
        MatrixAdd<<<dimGrid, dimBlock>>>(A, B, C_GPU);
    	cudaDeviceSynchronize();
        std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
        std::cout<<std::endl;
        if(b_print=='Y') C_GPU.Print();
        std::cout<<"Validation: "<<std::endl;
        Matrix::MatrixCompare(C_CPU,C_GPU);
        }
        break;
    case 1:
        {
        std::cout<<"Matrix subtraction: A-B"<<std::endl;
        std::cout<<std::endl;
        clock_t start = clock();
        Matrix C_CPU = A - B;
        std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
        std::cout<<std::endl;
        if(b_print=='Y') C_CPU.Print();


        std::cout<<std::endl;
        start = clock();
        MatrixSubtract<<<dimGrid, dimBlock>>>(A, B, C_GPU);
    	cudaDeviceSynchronize();
        std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
        std::cout<<std::endl;
        if(b_print=='Y') C_GPU.Print();
        std::cout<<"Validation: "<<std::endl;
        Matrix::MatrixCompare(C_CPU,C_GPU);
        }
        break;
    case 2:
        {
        std::cout<<"Matrix multiplication: A*B"<<std::endl;
        std::cout<<std::endl;
        clock_t start = clock();
        Matrix C_CPU = A * B;
        std::cout<<"CPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
        std::cout<<std::endl;
        if(b_print=='Y') C_CPU.Print();


        std::cout<<std::endl;
        start = clock();
        MatrixMultiply<<<dimGrid, dimBlock>>>(A, B, C_GPU);
	    cudaDeviceSynchronize();
        std::cout<<"GPU computing time [s]: "<< (clock() - start)/(1.0*CLOCKS_PER_SEC)<<std::endl;
        std::cout<<std::endl;
        if(b_print=='Y') C_GPU.Print();
        std::cout<<"Validation: "<<std::endl;
        Matrix::MatrixCompare(C_CPU,C_GPU);
        }
        break;

}












}
