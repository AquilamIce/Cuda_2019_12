#include "Matrix.h"
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>



//Matrix constructor
Matrix::Matrix(int width, int height)
{
	//Hide variables
	_width = width;
	_height = height;

	//CUDA memory
	cudaMallocManaged(&_elements, width * height * sizeof(float));
}

//Matrix copy constructor
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

//Matrix deconstructor
Matrix::~Matrix()
{
	cudaFree(_elements);
}

//Matrix initialization method
void Matrix::Initialize(float num, bool randomize)
{
	//Fill matrix with random numbers
	if(randomize)
	{
		for(int i = 0; i < _width * _height; i++)
		{
			_elements[i] = rand()/(float)RAND_MAX;; //Random number x:(0,1)
		}
	}
	//Fill matrix with one number
	else
	{
		for(int i = 0; i < _width * _height; i++)
		{
			_elements[i] = num;
		}
	}
}

//Matrix print method
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

//Matrix compare method
void Matrix::MatrixCompare(Matrix &A, Matrix &B)
{
	for(int row = 0; row < A._height; row++)
	{	for(int col = 0; col < A._width; col++)
		{
			if(fabs(A._elements[(row*A._width) + col] - B._elements[(row*B._width) + col]) > 1e-4)
			{
				fprintf(stderr, "Result verification failed at element M[%d;%d]!\t%f|%f\n", row+1, col+1, A._elements[(row*A._width) + col], B._elements[(row*B._width) + col]);
				break;
			}
		}
	}
	std::cout<<"Check ok"<<std::endl;
}

//Matrix + operator overload
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

//Matix - operator overload
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

//Matrix * operator overload
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

//Matrix = operator overload
const Matrix& Matrix::operator=(const Matrix &A)
{
  if (&A == this)
    return *this;
  cudaFree(_elements);
  _width = A._width;
  _height = A._height;

  cudaMallocManaged(&_elements, _width * _height * sizeof(float));

  for(int i = 0; i < A._width * A._height; ++i)
  {
	  this->_elements[i] = A._elements[i];
  }

  return *this;
}

//Matrix transpose method
Matrix Matrix::Transpose()
{
	Matrix result = Matrix(_height, _width);
	for (int row = 0; row < result._height; row++)
		for (int col = 0; col < result._width; col++)
			result._elements[row*result._width + col] = _elements[col*_width + row];
	return result;
}

//Matrix elements summation method
float Matrix::VectorSum()
{
	float result = 0;

	for(int i = 0; i < _width * _height; i++)
	{
		result += _elements[i];
	}
	return result;
}

//Matrix [] operator overload
float& Matrix::operator[](int i)
{
	return _elements[i];
}

const float& Matrix::operator[](int i) const
{
	return _elements[i];
}
