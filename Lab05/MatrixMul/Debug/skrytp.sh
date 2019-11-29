#!/bin/bash

echo "Script started..."

nvprof ./MatrixMultiplicationFinal 2>&1 | tee >> per.txt
nvprof ./MatrixMultiplicationFinal 2>&1 | tee >> per.txt
nvprof ./MatrixMultiplicationFinal 2>&1 | tee >> per.txt
nvprof ./MatrixMultiplicationFinal 2>&1 | tee >> per.txt
nvprof ./MatrixMultiplicationFinal 2>&1 | tee >> per.txt

echo >> ./per.txt
echo >> ./per.txt
echo >> ./per.txt
echo >> ./per.txt
echo >> ./per.txt

echo "Script finished..."
