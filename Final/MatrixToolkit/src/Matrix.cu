#include <stdio.h>
#include "Matrix.h"
#include <stdlib.h>
#include <iostream>
#include <cmath>



Matrix::Matrix(int width, int height)
{
	_width = width;
	_height = height;

	cudaMallocManaged(&_elements, width * height * sizeof(float));
}

Matrix::~Matrix()
{
	cudaFree(_elements);
}

/*
Matrix::Matrix(const Matrix &matrix)
{
	_width = matrix._width;
	_height = matrix._height;
	cudaMallocManaged(&_elements, _width * _height * sizeof(float));
		for(int i = 0; i < matrix._width * matrix._height; i++)
		{
			_elements[i] = matrix._elements[i];
		}
}
*/

void Matrix::Initialize(float num, bool randomize)
{
	if(randomize)
	{

	}
	else 
	{
		for(int i = 0; i < _width * _height; i++)
		{
			_elements[i] = num;
		}
	} 
}

void Matrix::Initialize(const float* num)
{
		for(int i = 0; i < _width * _height; i++)
		{
			_elements[i] = num[i];
		}

}

void Matrix::Print()
{
	for(int row = 0; row < _height; row ++)
	{
		for(int col = 0; col < _width; col++)
		{
			printf("M[%d;%d] = %f\t", row + 1, col + 1, _elements[(row*_width) + col]);
		}
		printf("\n");
	}
}

void Matrix::MatrixCompare(Matrix &A, Matrix &B)
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


Matrix Matrix::operator+(const Matrix &matrix)
{

	if (matrix._width != _width || matrix._height != _height)
	{
		fprintf(stderr, "Wrong matrix size");
		exit(EXIT_FAILURE);
	}

	Matrix result = Matrix(_width, _height);

	 for (int i = 0; i < _width * _height; i++)
	 {
	 	result._elements[i] = this->_elements[i] + matrix._elements[i];
	 }

	return result;
}

Matrix Matrix::operator-(const Matrix &matrix)
{
	if (matrix._width != _width || matrix._height != _height)
	{
		fprintf(stderr, "Wrong matrix size");
		exit(EXIT_FAILURE);
	}

	Matrix result = Matrix(_width, _height);

	for (int i = 0; i < _width * _height; i++)
	{
		result._elements[i] = this->_elements[i] - matrix._elements[i];
	}

	return result;
}

Matrix Matrix::operator*(const Matrix &matrix)
{
	if (_width != matrix._height)
	{
		fprintf(stderr, "wrong matrix size");
		exit(EXIT_FAILURE);
	}

	Matrix result = Matrix(matrix._width, _height);

	for(int row = 0; row < result._height; row++)
	{
		for(int col = 0; col < result._width; col++)
		{
			for(int k = 0; k < _width; k++)
				result._elements[(row*result._width) + col] += _elements[(row*_width) + k] * matrix._elements[col + (k*matrix._width)];
		}
	}

	return result;
}


float& Matrix::operator[](int i)
{
	return _elements[i];
}

const float& Matrix::operator[](int i) const
{
	return _elements[i];
}

