#pragma once

#include <iostream>

class Matrix
{
public:
	int _width;
	int _height;
	float* _elements;

	//Constructor
	Matrix(int width, int height);
	Matrix(const Matrix &matrix);

	//Deconstructor
	~Matrix();

	//Initialize Matrix
	void Initialize(float num = 0, bool randomize = false);
	//Print Matrix
	void Print();


	//Operators
	Matrix operator+(const Matrix &matrix);
	Matrix operator-(const Matrix &matrix);
	Matrix operator*(const Matrix &matrix);
	const Matrix& operator=(const Matrix &matrix);
	float& operator[](int i);
	const float& operator[](int i) const;

	Matrix Transpose();
	float VectorSum();
	//Compare two matrices
	static void MatrixCompare(Matrix &A, Matrix &B);

private:

};
