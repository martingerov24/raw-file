#include "CudaClass.h"

__device__ __forceinline__
uint8_t Color(uint16_t number)
{
	uint8_t n;
	// this is the solution if little indian
	uint8_t first_8_bits = number & 0b11111111; first_8_bits = first_8_bits >> 4;
	number = number >> 8;
	n = number & 0b11111111; n = n >> 4;
	// so now we have smth like bit1 -> 10101011, bit0 -> 10101011;
	n = n & 0b1111; // basicly the paddings are throun away
	// now we have 2 4 bit numbers and when combining them OR || XOR
	n = n << 4;
	n |= first_8_bits;
	// the second number is our putput
	return n;
}

__global__
void Checker(uint16_t* __restrict__ d_Data, uint8_t* __restrict__ cpy_Data, int width, int height)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < width && x >= 0
		&& y < height && y >= 0)
	{
		int calc = y * width + x;  //their scope is threadLifeTime
		uint8_t n = Color(d_Data[calc]);
		//h   !w
		short idx = (y & 1) + !(x & 1);
		uint8_t rgb[3] = { 0,0,0 };
		rgb[idx] = n;

		cpy_Data[3 * calc + 0] = rgb[0];
		cpy_Data[3 * calc + 1] = rgb[1];
		cpy_Data[3 * calc + 2] = rgb[2];
	}
}

void Cuda::rawValue()
{
	//auto start = std::chrono::high_resolution_clock::now();
	cudaStatus = cudaMemcpyAsync(d_data, data.data(), sizeof(uint16_t) * size, cudaMemcpyHostToDevice, stream);
	assert(cudaStatus == cudaSuccess && "not able to tansfer Data!");

	dim3 sizeOfBlock(((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), height);

	Checker << <sizeOfBlock, THREADS_PER_BLOCK, 0, stream >> > (d_data, cpyData, width, height);
	auto status = cudaGetLastError();

	cudaStatus = cudaMemcpyAsync(h_cpy.data(), cpyData, sizeof(uint8_t) * size * 3, cudaMemcpyDeviceToHost, stream);
	assert(cudaStatus == cudaSuccess && "not able to transfer device to host!");
}