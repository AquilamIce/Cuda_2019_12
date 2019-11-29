#include "Matrix.h"
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

Matrix::Matrix(int width, int height)
{
	_width = width;
	_height = height;
	_elements = new float[width*height];
	_type = CPU;
	for(int i =0; i < width * height; i++)
	{
		_elements[i] = 0;
	}
}

Matrix::Matrix(int width, int height, MemoryType type)
{
	_width = width;
	_height = height;
	_type = type;
	cudaMallocManaged(&_elements, width * height * sizeof(float));
	for(int i =0; i < width * height; i++)
	{
		_elements[i] = 0;
	}
}

Matrix::~Matrix()
{
	switch (_type)
	{
	case CPU:
		delete[] _elements;
		break;
	case cuda:
		cudaFree(_elements);
		break;
	}
}
