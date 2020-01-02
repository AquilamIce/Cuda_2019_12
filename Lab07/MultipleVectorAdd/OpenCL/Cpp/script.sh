#!/bin/bash

echo "Script started..."

for(( i=0; i<30; i++))
do
	echo "Computing for... 2^$i "
	./vadd $((2**$i)) 2>&1 | tee >> per.txt

	echo >> ./per.txt
	echo >> ./per.txt
	echo >> ./per.txt
	echo >> ./per.txt
	echo >> ./per.txt
	echo "Done.."
done

echo "Script finished..."
