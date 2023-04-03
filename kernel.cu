/*
* This program will do a load and store operation on each element of a vector.
* The access to the vector is strided (where stride 1 = coalesced).
* It meassures the bandwidth in GB/s for different stride sizes and on
* CPU and GPU.
*
* +---------+                        +---------+
* |111111111| + 1 (on each Thread) = |222222222|
* +---------+                        +---------+
*
* vector   = all Ones
*
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <chrono>

using namespace std;


/* This is our CUDA call wrapper, we will use in PAC.
*
*  Almost all CUDA calls should be wrapped with this makro.
*  Errors from these calls will be catched and printed on the console.
*  If an error appears, the program will terminate.
*
* Example: gpuErrCheck(cudaMalloc(&deviceA, N * sizeof(int)));
*          gpuErrCheck(cudaMemcpy(deviceA, hostA, N * sizeof(int), cudaMemcpyHostToDevice));
*/
#define GPU_ERR_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(const cudaError_t code, const char* file, const int line, const bool abort = true)
{
	if (code == cudaSuccess) return;

	std::cout << "GPU assert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
	if (abort) exit(code);
}

// GPU kernel which access an vector with a stride pattern
__global__ void strided_kernel(int* const vec, const int size, const int stride)
{
	const auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride % size;
	vec[idx] += 1;
}

// Execute a loop of different strides accessing a vector as GPU kernels.
// Measure the spent time and print out the reached bandwidth in GB/s.
void gpu_stride_loop(int* const device_vec, const int size)
{
	// Define some helper values
	const int processed_mb = size * sizeof(int) / 1024 / 1024 * 2; // 2x as 1 read and 1 write
	constexpr int block_size = 256;
	float ms;

	// Init CUDA events used to meassure timings 
	cudaEvent_t start_event, stop_event;
	GPU_ERR_CHECK(cudaEventCreate(&start_event))
	GPU_ERR_CHECK(cudaEventCreate(&stop_event))

	// Warm up GPU (The first kernel of a program has more overhead than the followings)
	GPU_ERR_CHECK(cudaEventRecord(start_event, nullptr))
	strided_kernel<<<size / block_size, block_size>>>(device_vec, size, 1);
	GPU_ERR_CHECK(cudaEventRecord(stop_event, nullptr))
	GPU_ERR_CHECK(cudaEventSynchronize(stop_event))

	GPU_ERR_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event))
	cout << "GPU warmup kernel: " << processed_mb / ms << "GB/s bandwidth" << endl;

	for (int stride = 1; stride <= 32; ++stride)
	{
		GPU_ERR_CHECK(cudaEventRecord(start_event, nullptr))
		strided_kernel<<<size / block_size, block_size>>>(device_vec, size, stride);
		GPU_ERR_CHECK(cudaEventRecord(stop_event, nullptr))
		GPU_ERR_CHECK(cudaEventSynchronize(stop_event))

		GPU_ERR_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event))
		cout << "GPU stride size " << stride << ": " << processed_mb / ms << "GB/s bandwidth" << endl;
	}
}

// Execute a loop of different strides accessing a vector.
// Measure the spent time and print out the reached bandwidth in GB/s.
void cpu_stride_loop(int* const vec, const int size)
{
	const float processed_mb = size * sizeof(int) / 1024 / 1024 * 2; // 2x as 1 read and 1 write
	for (int stride = 1; stride <= 32; stride++)
	{
		auto start = chrono::high_resolution_clock::now();

		for (int i = 0; i < size; i++)
		{
			int strided_i = (i * stride) % size;
			vec[strided_i] = vec[strided_i] + 1;
		}

		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
		cout << "CPU stride size " << stride << ": " << processed_mb / duration.count() << "GB/s bandwidth" << endl;
	}
}

// Init und destruct memory and call the CPU and the GPU measurement code.
int main()
{
	// Define the size of the vector in MB
	constexpr int width_mb = 128;
	constexpr int width = width_mb * 1024 * 1024 / sizeof(int);

	// Allocate and prepare input vector
	const auto host_vector = new int[width];
	for (int index = 0; index < width; index++)
	{
		host_vector[index] = 1;
	}

	// Allocate device memory
	int* device_vector;
	GPU_ERR_CHECK(cudaMalloc(&device_vector, width * sizeof(int)))

	// Copy data from host to device
	GPU_ERR_CHECK(cudaMemcpy(device_vector, host_vector, width * sizeof(int), cudaMemcpyHostToDevice))

	// run stride loop on CPU to have some reference values
	cpu_stride_loop(host_vector, width);
	cout << "--------------------------------------------------------" << endl;

	// run stride loop on GPU
	gpu_stride_loop(device_vector, width);

	// Free memory on device & host
	cudaFree(device_vector);
	delete[] host_vector;

	return 0;
}
