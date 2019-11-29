#pragma once

#include <iostream>

enum MemoryType
{
	cuda,
	CPU
};

class Matrix
{
public:
	int _width;
	int _height;
	float* _elements;

	Matrix(int width, int height);
	Matrix(int width, int height, MemoryType type);
	~Matrix();
private:
	MemoryType _type;
};


