#define THREADS_PER_BLOCK 1024
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <string>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//------------------------------------------------Predifining Functions
std::vector<uint16_t> ReadingFiles(char* fileName, int& height, int& width);
void outPutFile(std::vector<uint8_t> pixelData, int height, int width);
//------------------------------------------------Color operations
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

__global__ void Checker(uint16_t* d_Data, uint8_t* cpy_Data, int width, int height)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y; 
	if (x < width && x >= 0 
		&& y < height && y>=0)
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

__host__
void GetCudaRdy(std::vector<uint8_t> &h_cpy, const std::vector<uint16_t>& data, const int& height, const int& width)
{
	uint16_t* d_data;
	cudaError_t cudaStatus = cudaError_t(0);
	int size = height * width;
	//uint8_t* h_cpy = new uint8_t[width * height * 3];
	uint8_t* cpyData;

	cudaStatus = cudaSetDevice(0);
	assert(cudaStatus == cudaSuccess, "you do not have cuda capable device!");

	cudaStatus = cudaMalloc((void**)&d_data, data.size() * sizeof(uint16_t));//sizeof(uint16_t) * size
	assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&cpyData, sizeof(uint8_t) * h_cpy.size() * 3);
	assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");
	// learn to get stram and make the function async
	auto start = std::chrono::high_resolution_clock::now();
	
	cudaStatus = cudaMemcpyAsync(d_data, data.data(), sizeof(uint16_t) * size, cudaMemcpyHostToDevice);
	assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");

	cudaStatus = cudaMemcpyAsync(cpyData, h_cpy.data(), sizeof(uint8_t) * size * 3, cudaMemcpyHostToDevice);
	assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");// here i am actually not in need to transfer data, but i wanted to see if it makes difference
	dim3 sizeOfBlock(((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), height); // 4 , 2

	Checker << <sizeOfBlock, THREADS_PER_BLOCK >> > (d_data, cpyData, width, height);

	cudaStatus = cudaMemcpyAsync(h_cpy.data(), cpyData, sizeof(uint8_t) * size * 3, cudaMemcpyDeviceToHost);
	//cudaStatus = cudaStreamSynchronize();
	cudaStatus = cudaDeviceSynchronize();
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	printf("%d -> is the time measured", duration);

	cudaFree(cpyData);
	cudaFree(d_data);// it was said to -> cudaFree ( void* devPtr )Frees memory on the device.
	return;
}

bool IfFileCanOpen(const char* fileName, const std::string width, const std::string height) {
	FILE* file = fopen(fileName, "r");
	if (file == 0)
	{
		printf("not able to open the file for reading!\n");
		return false;
	}
	fclose(file);
	for (int i = 0; i < fmax(width.size(), height.size()); i++)
	{
		if (width[i] > '9' || width[i] < '0'
			|| height[i] > '9' || height[i] < '0')
		{
			printf("you have declared invalid numbers for width and height\n");
			return false;
		}
	}
	return true;
}
int main(int argc, char** argv)
{
	char* fileName = "fileToRead.raw";
	int width = 3840, height = 1920;
	const std::vector<uint16_t> &data = ReadingFiles(fileName, height, width);// reading the 

	std::vector<uint8_t> h_cpy;
	h_cpy.resize(width * height * 3);

	GetCudaRdy(h_cpy, data, height, width);
	outPutFile(h_cpy, height, width);// ofstreaming the files
	return 0;
}

std::vector<uint16_t> ReadingFiles(char* fileName, int& height, int& width)
{
	FILE* rdFile = fopen(fileName, "rb+");
	std::vector<uint16_t> data;
	if (rdFile == 0) {
		printf("no file found!");
		return data; 
	}
	data.resize(height * width);
	fread(reinterpret_cast<char*>(&data[0]), 2, height * width, rdFile);
	fclose(rdFile);
	return data;
}
void outPutFile(std::vector<uint8_t> pixelData, int height, int width)
{
	FILE* fileWr;
	fileWr = fopen("writingFile.ppm", "w+");
	fprintf(fileWr, "%s %d %d %d ", "P6", width, height, 255);
	fclose(fileWr);

	fileWr = fopen("writingFile.ppm", "ab+");
	fwrite(reinterpret_cast<const char*>(&pixelData[0]), 1, sizeof(uint8_t) * width * height * 3, fileWr);
	fclose(fileWr);
	fileWr = nullptr;
}