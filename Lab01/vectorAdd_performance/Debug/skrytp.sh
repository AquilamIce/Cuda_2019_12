#!/bin/bash

echo "Script started..."

nvprof ./vectorAdd_performance 2>&1 | tee >> per.txt
nvprof ./vectorAdd_performance 2>&1 | tee >> per.txt
nvprof ./vectorAdd_performance 2>&1 | tee >> per.txt
nvprof ./vectorAdd_performance 2>&1 | tee >> per.txt
nvprof ./vectorAdd_performance 2>&1 | tee >> per.txt

echo >> ./per.txt
echo >> ./per.txt
echo >> ./per.txt
echo >> ./per.txt
echo >> ./per.txt

echo "Script finished..."
