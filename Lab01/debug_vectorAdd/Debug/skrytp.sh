#!/bin/bash

echo "Script started..."

nvprof ./debug_vectorAdd 2>&1 | tee >> per.txt
nvprof ./debug_vectorAdd 2>&1 | tee >> per.txt
nvprof ./debug_vectorAdd 2>&1 | tee >> per.txt
nvprof ./debug_vectorAdd 2>&1 | tee >> per.txt
nvprof ./debug_vectorAdd 2>&1 | tee >> per.txt

echo >> ./per.txt
echo >> ./per.txt
echo >> ./per.txt
echo >> ./per.txt
echo >> ./per.txt

echo "Script finished..."
