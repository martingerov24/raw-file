#pragma once
#define THREADS_PER_BLOCK 32
#include "cuda_runtime.h"
#include "cuda/std/cmath"
#include "device_launch_parameters.h"
#include <vector>
#include <cassert>
#include <chrono>


class CudaKeypoints
{
public:
	CudaKeypoints(std::vector<uint8_t>& h_result, const float2* keypoints, const std::vector<uint8_t>& data,
		const int height, const int width, uint8_t* d_image, uint8_t* d_result, float2* d_kp, int sizeOfKeypoints)
		
		: h_result(h_result), keypoints(keypoints)
		, img(data), height(height)
		, width(width), d_image(d_image)
		, d_result(d_result), d_kp(d_kp)
		, sizeofKeypoints(sizeOfKeypoints)
	{
		cudaStatus = cudaError_t(0);
		size = height * width;
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		cudaStatus = cudaSetDevice(0);
		assert(cudaStatus == cudaSuccess, "you do not have cuda capable device!");
		cudaStatus = cudaStreamCreate(&stream);
	}

	__host__
		void startup(int size)
	{
		cudaStatus = cudaMalloc((void**)&d_image, img.size() * sizeof(uint8_t));//sizeof(uint16_t) * size
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");

		cudaStatus = cudaMalloc((void**)&d_result, sizeof(uint8_t) * h_result.size());
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");

		cudaStatus = cudaMalloc((void**)&d_kp, sizeof(float) * sizeofKeypoints);
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");
	}

	__host__
	void Kernel();

	__host__
		void cudaKernel()
	{
		cudaStatus = cudaMemcpyAsync(d_image, img.data(), sizeof(uint8_t) * size, cudaMemcpyHostToDevice, stream);
		assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");

		cudaStatus = cudaMemcpyAsync(d_kp, keypoints, sizeof(float2) * size, cudaMemcpyHostToDevice, stream);
		assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");

		Kernel();
	}
	__host__
		void cudaMemcpyD2H()
	{
		cudaStatus = cudaMemcpyAsync(h_result.data(), d_result, sizeof(uint8_t) * size, cudaMemcpyDeviceToHost, stream);
		assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");
	}
	__host__
		void sync()
	{
		cudaStatus = cudaStreamSynchronize(stream);
	}

	__host__
		~CudaKeypoints()
	{
		cudaFree(d_image);
		cudaFree(d_result);// it was said to -> cudaFree ( void* devPtr )Frees memory on the device.
		cudaFree(d_kp);
		cudaStreamDestroy(stream);
	}
private:
	//-------Provided By Main-------
	std::vector<uint8_t>& h_result;
	const float2* &keypoints;
	const std::vector<uint8_t>& img;
	const int height, width;
	int size; int sizeofKeypoints;
	//------Provided By Class,------
	//should be deleted after using
	uint8_t* &d_image;
	uint8_t* &d_result;
	float2* &d_kp;
	cudaStream_t stream;
	cudaError_t cudaStatus;
};






class Cuda
{
public:
	Cuda(std::vector<uint8_t>& h_cpy, const std::vector<uint16_t>& data,
		const int& height, const int& width,
		uint16_t*& d_data, uint8_t*& cpyData)
		:h_cpy(h_cpy), data(data), height(height), width(width)
		, d_data(d_data), cpyData(cpyData)
	{
		cudaStatus = cudaError_t(0);
		size = height * width;
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		cudaStatus = cudaSetDevice(0);
		assert(cudaStatus == cudaSuccess, "you do not have cuda capable device!");
		cudaStatus = cudaStreamCreate(&stream);
	}
	
	__host__
		void startup(int size)
	{
		cudaStatus = cudaMalloc((void**)&d_data, size * sizeof(uint16_t));//sizeof(uint16_t) * size
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");

		cudaStatus = cudaMalloc((void**)&cpyData, sizeof(uint8_t) * size * 3);
		assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");
	}
	__host__
	void rawValue();

	__host__
	void sync()
	{
		cudaStatus = cudaStreamSynchronize(stream);
	}
	void outPutFile()
	{
		FILE* fileWr;
		fileWr = fopen("writingFile.ppm", "w+");
		fprintf(fileWr, "%s %d %d %d ", "P6", width, height, 255);
		fclose(fileWr);

		fileWr = fopen("writingFile.ppm", "ab+");
		fwrite(reinterpret_cast<const char*>(&h_cpy[0]), 1, sizeof(uint8_t) * width * height * 3, fileWr);
		fclose(fileWr);
		fileWr = nullptr;
	}
	__host__
		~Cuda()
	{
		cudaFree(cpyData);
		cudaFree(d_data);// it was said to -> cudaFree ( void* devPtr )Frees memory on the device.
		cudaStreamDestroy(stream);
	}
protected:
	int height, width, size;
	uint16_t* &d_data;
	uint8_t* &cpyData;
	std::vector<uint8_t> &h_cpy;
	const std::vector<uint16_t> &data;
	cudaStream_t stream;
	cudaError_t cudaStatus;
};
