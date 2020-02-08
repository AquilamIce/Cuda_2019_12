#include "CudaFunctions.h"

#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "Matrix.h"


class MatrixCalculator
{
public:
	MatrixCalculator();

	//Perform Matrix operations to find difference in performance between GPU and CPU algorithms.
	void MatrixAddition();
	void MatrixSubtraction();
	void MatrixMultiplication();
	void MatrixTransposition();
	void VectorSummation();

private:
	Matrix CreateRandomMatrix(int width, int height);
	int deviceId;
	int SMCount;
};
