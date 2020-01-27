#include "CudaFunctions.h"

#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "Matrix.h"

class MatrixCalculator
{
private:
	Matrix CreateRandomMatrix(int width, int height);

public:

	void MatrixAddition();
	void MatrixSubtraction();
	void MatrixMultiplication();
	void MatrixTransposition();
	void VectorSummation();
};
