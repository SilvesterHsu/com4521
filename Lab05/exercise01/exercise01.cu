#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 65536
#define THREADS_PER_BLOCK 128

#define PUMP_RATE 2

#define READ_BYTES N*(2*4)  //2 reads of 4 bytes (a and b)
#define WRITE_BYTES N*(4*1) //1 write of 4 bytes (to c)

void checkCUDAError(const char*);
void random_ints(int* a);

__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];

__global__ void vectorAdd() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_c[i] = d_a[i] + d_b[i];
}



int main(void) {
	int* a, * b, * c, * c_ref;			// host copies of a, b, c
	//int* d_a, * d_b, * d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * sizeof(int);
	cudaEvent_t start, stop;
	float milliseconds = 0;
	int device_count;
	double theoretical_BW;
	double measure_BW;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaGetDeviceCount(&device_count);
	if (device_count > 0) {
		cudaSetDevice(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		theoretical_BW = deviceProp.memoryClockRate * PUMP_RATE * (deviceProp.memoryBusWidth / 8.0) / 1e6; //convert to GB/s
	}

	// Alloc space for device copies of a, b, c
	//cudaMalloc((void**)&d_a, size);
	//cudaMalloc((void**)&d_b, size);
	//cudaMalloc((void**)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int*)malloc(size); random_ints(a);
	b = (int*)malloc(size); random_ints(b);
	c = (int*)malloc(size);
	c_ref = (int*)malloc(size);

	// Copy inputs to device
	//cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_a, a, size);
	cudaMemcpyToSymbol(d_b, b, size);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
	cudaEventRecord(start);
	vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > ();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	measure_BW = (READ_BYTES + WRITE_BYTES) / (milliseconds * 1e6);
	checkCUDAError("CUDA kernel");


	// Copy result back to host
	//cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(c, d_c, size);
	checkCUDAError("CUDA memcpy");

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	//checkCUDAError("CUDA cleanup");

	printf("Execution time is %f ms\n", milliseconds);
	printf("Theoretical Bandwidth is %f GB/s\n", theoretical_BW);
	printf("Measured Bandwidth is %f GB/s\n", measure_BW);

	return 0;
}

void checkCUDAError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int* a)
{
	for (unsigned int i = 0; i < N; i++) {
		a[i] = rand();
	}
}