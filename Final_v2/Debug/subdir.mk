################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../CudaFunctions.cu \
../Function.cu \
../Main.cu \
../Matrix.cu \
../MatrixCalculator.cu 

OBJS += \
./CudaFunctions.o \
./Function.o \
./Main.o \
./Matrix.o \
./MatrixCalculator.o 

CU_DEPS += \
./CudaFunctions.d \
./Function.d \
./Main.d \
./Matrix.d \
./MatrixCalculator.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


