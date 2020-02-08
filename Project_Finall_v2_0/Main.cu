#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include <cuda_runtime.h>

#include "Matrix.h"
#include "CudaFunctions.h"
#include "MatrixCalculator.h"


int main(int argc, char **argv)
{
	MatrixCalculator Calculator; //Matrix Calculator manages all operations on matrices

	std::cout<<"DEMO OF COMPUTING PROGRAM"<<std::endl;


	//Simple interface to set all parameters and manage program
	char operation = '1';
	while (operation != '0')
	{
		std::cout<<"Choose operation: \n"
			"0 - Exit\n"
			"1 - Addition\n"
			"2 - Subtraction\n"
			"3 - Multiplication\n"
			"4 - Transposition\n"
			"5 - VectorSum\n"
			<<std::endl;

		if (!std::cin)
	    {
	        std::cin.clear();
	        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	    }

		//Get information from user
		std::cin>>operation;
		std::cout<<std::endl;

		//Case statement to menu management
		switch(operation)
		{
		case '0':
			break;
		case '1':
			Calculator.MatrixAddition();
			break;
		case '2':
			Calculator.MatrixSubtraction();
			break;
		case '3':
			Calculator.MatrixMultiplication();
			break;
		case '4':
			Calculator.MatrixTransposition();
			break;
		case '5':
			Calculator.VectorSummation();
			break;
		default:
			printf("Wrong input\n");
		}
	}
}
