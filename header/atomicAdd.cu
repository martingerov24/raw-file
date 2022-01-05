#include "../header/CudaClass.h"

__global__ void sumKernel(float2* __restrict__ arr,
	float2* output, float exclude)
{
	int numPoints = arr[0].x;//*(int*)(&arr[0].x);
	int* idx = (int*)(&output[0].x);
	int gthread_id = blockIdx.x * blockDim.x + threadIdx.x + 1;

	if (gthread_id >= numPoints)// well i don't need to save the first float of the array 
	{							 // and the only elements this will work on are odd
		return;
	}

	float2 element = arr[gthread_id];

	if (element.y >= exclude) {
		return;
	}

	int asd = atomicAdd(idx, 1);
	//counter++; // this element refers to the atom Int
	output[asd+1] = element;
}

__host__ void CudaKeypoints::getSmallElements(std::vector<float2> &arr, std::vector<float2>& output, float exclude, cudaStream_t& providedStream)
{
	float2* d_arr;
	float2* d_output;
	cudaStatus = cudaMallocAsync((void**)&d_arr, sizeof(float2) * arr.size(), providedStream);
	assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");

	cudaStatus = cudaMallocAsync((void**)&d_output, sizeof(float2) * arr.size(), providedStream);
	assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");

	cudaStatus = cudaMemcpyAsync(d_arr, arr.data(), sizeof(float2) * arr.size(), cudaMemcpyHostToDevice, providedStream);
	assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");

	const int numBlocks = (arr.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	sumKernel << <numBlocks, THREADS_PER_BLOCK, 0, providedStream >> > (d_arr, d_output, 2);
	auto cudaStatus = cudaGetLastError();
	assert(cudaStatus == cudaSuccess && "porblem with mathc kernel");

	cudaStatus = cudaMemcpyAsync(output.data(), d_output, sizeof(float2) * arr.size(), cudaMemcpyDeviceToHost, providedStream);

	cudaStatus = cudaStreamSynchronize(providedStream);
	assert(cudaStatus == cudaSuccess && "Failed to sync");

	cudaFreeAsync(d_arr, providedStream);
	cudaFreeAsync(d_output, providedStream);
}