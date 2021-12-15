#include "CudaClass.h"

__device__
void Color(uint16_t number, uint8_t& n)
{
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
}

__global__
void Checker(uint16_t* d_Data, uint8_t* cpy_Data, int width, int height)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < width && x >= 0
		&& y < height && y >= 0)
	{
		int calc = y * width + x;  //their scope is threadLifeTime
		uint8_t n = 0;
		Color(d_Data[calc], n);
		//h   !w
		short idx = (y & 1) + !(x & 1);
		cpy_Data[3 * calc + 0] = 0;//r
		cpy_Data[3 * calc + 1] = 0;//g
		cpy_Data[3 * calc + 2] = 0;//b
		cpy_Data[3 * calc + idx] = n;
	}
}

void Cuda::rawValue()
{
	//auto start = std::chrono::high_resolution_clock::now();
	cudaStatus = cudaMemcpyAsync(d_data, data.data(), sizeof(uint16_t) * size, cudaMemcpyHostToDevice, stream);
	assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");

	cudaStatus = cudaMemcpyAsync(cpyData, h_cpy.data(), sizeof(uint8_t) * size * 3, cudaMemcpyHostToDevice, stream);
	assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");// here i am actually not in need to transfer data, but i wanted to see if it makes difference
	dim3 sizeOfBlock(((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), height); // 4 , 2

	Checker << <sizeOfBlock, THREADS_PER_BLOCK, 0, stream >> > (d_data, cpyData, width, height);

	cudaStatus = cudaMemcpyAsync(h_cpy.data(), cpyData, sizeof(uint8_t) * size * 3, cudaMemcpyDeviceToHost, stream);

	/*auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	printf("%d -> is it working?", duration);*/
}