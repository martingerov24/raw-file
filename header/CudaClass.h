#pragma once
#define THREADS_PER_BLOCK 1024
#include "cuda_runtime.h"
#include "cuda/std/cmath"
#include "device_launch_parameters.h"
#include "nvtx3/nvToolsExt.h"

#include <vector>
#include <cassert>
#include <chrono>

extern std::vector<float> leftPoint;
extern std::vector<float> rightPoint;
extern std::vector<float> keypoints;

struct descriptor_t {
    uint64_t bits[4];
};

class CudaKeypoints
{
public:
    CudaKeypoints(const std::vector<uint8_t>& data,
        const int height, const int width)
        
        : img(data), height(height)
        , width(width), d_image(nullptr)
        , d_result(nullptr), d_kp(nullptr)
        , d_query(nullptr) , d_train(nullptr)
        , d_resMatcher(nullptr)
    {
        cudaStatus = cudaError_t(0);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        cudaStatus = cudaSetDevice(0);
        assert(cudaStatus == cudaSuccess && "you do not have cuda capable device!");
        cudaStatus = cudaStreamCreate(&stream);
    }

    __host__ void startup(int size, int providedKPsize)
    {
        cudaStatus = cudaMalloc((void**)&d_image, sizeof(uint8_t) * size);
        assert(cudaStatus == cudaSuccess && "cudaMalloc failed!");

        cudaStatus = cudaMalloc((void**)&d_result, sizeof(descriptor_t) * providedKPsize); 
        assert(cudaStatus == cudaSuccess && "cudaMalloc failed!");

        cudaStatus = cudaMalloc((void**)&d_kp, sizeof(float2) * providedKPsize);
        assert(cudaStatus == cudaSuccess && "cudaMalloc failed!");
    }

    __host__ void MemoryAllocationManagedForMatches(int querySize, int trainSize)
    {
        cudaStatus = cudaMallocManaged((void**)&d_query, sizeof(descriptor_t) * querySize);
        assert(cudaStatus == cudaSuccess && "not able to allocate memory on device1");
        
		cudaStatus = cudaMallocManaged((void**)&d_train, sizeof(descriptor_t) * trainSize);
        assert(cudaStatus == cudaSuccess && "not able to allocate memory on device2");

		cudaStatus = cudaMallocManaged((void**)&d_resMatcher, sizeof(uint16_t) * querySize);
        assert(cudaStatus == cudaSuccess && "not able to allocate memory on device3");
    }

    __host__ void AttachMemAsync(cudaStream_t& providedstream, std::vector<descriptor_t>& query, std::vector<descriptor_t>& train)
    {
        cudaStatus = cudaStreamAttachMemAsync(providedstream, d_query, sizeof(descriptor_t) * query.size(), cudaMemAttachGlobal);
        assert(cudaStatus == cudaSuccess && "not able to attach memory on ram -> for gd sake");

        cudaStatus = cudaStreamAttachMemAsync(providedstream, d_train, sizeof(descriptor_t) * train.size(), cudaMemAttachGlobal);
        assert(cudaStatus == cudaSuccess && "assert in attach memory on ram.... for second time");

		cudaStatus = cudaStreamAttachMemAsync(providedstream, d_resMatcher, sizeof(uint16_t) * query.size(), cudaMemAttachGlobal);
		assert(cudaStatus == cudaSuccess && "assert in attach memory on ram.... for second time");

		cudaStatus = cudaMemcpyAsync(d_query, query.data(), sizeof(descriptor_t) * query.size(), cudaMemcpyHostToDevice, providedstream);
		cudaStatus = cudaMemcpyAsync(d_train, train.data(), sizeof(descriptor_t) * train.size(), cudaMemcpyHostToDevice, providedstream);

		cudaStatus = cudaStreamSynchronize(providedstream);
        assert(cudaStatus == cudaSuccess && "not able to sync after memory attachment");
    }
    __host__ void MemoryAllocationAsync(cudaStream_t &providedstream , int querysize, int trainsize)
    {
        cudaStatus = cudaMallocAsync((void**)&d_query, sizeof(descriptor_t) * querysize,  providedstream);
        assert(cudaStatus == cudaSuccess && "not able to allocate memory on device1");
        cudaStatus = cudaMallocAsync((void**)&d_train, sizeof(descriptor_t) *  trainsize,  providedstream);
        assert(cudaStatus == cudaSuccess && "not able to allocate memory on device2");
        cudaStatus = cudaMallocAsync((void**)&d_resMatcher, sizeof(uint16_t) * querysize,  providedstream);
        assert(cudaStatus == cudaSuccess && "not able to allocate memory on device3");
    }

    __host__ void cudaUploadKeypoints(std::vector<float>& kp)
    {
        cudaStatus = cudaMemcpyAsync(d_image, img.data(), sizeof(uint8_t) * height * width, cudaMemcpyHostToDevice, stream);
        assert(cudaStatus == cudaSuccess && "not able to tansfer Data!");

        cudaStatus = cudaMemcpyAsync(d_kp, kp.data(), sizeof(float) * kp.size(), cudaMemcpyHostToDevice, stream);
        assert(cudaStatus == cudaSuccess && "not able to tansfer Data!");
    }

    __host__ void MemcpyUploadAsyncForMatches(cudaStream_t& stream2, std::vector<descriptor_t>& query, std::vector<descriptor_t>& train)
    {
        cudaStatus = cudaMemcpyAsync(d_query, query.data(), sizeof(descriptor_t) * query.size(), cudaMemcpyHostToDevice, stream2);
        assert(cudaStatus == cudaSuccess && "assert in cudaMemcpyAsync 1");
        cudaStatus = cudaMemcpyAsync(d_train, train.data(), sizeof(descriptor_t) * train.size(), cudaMemcpyHostToDevice, stream2);
        assert(cudaStatus == cudaSuccess && "assert in cudaMemcpyAsync 2");
    }

    __host__ void downloadAsync(cudaStream_t &provided_stream, std::vector<uint16_t> &result, int size)
    {
        result.resize(size);
        cudaStatus = cudaMemcpyAsync(result.data(), d_resMatcher, sizeof(uint16_t) * size, cudaMemcpyDeviceToHost, provided_stream);
        assert(cudaStatus == cudaSuccess && "download Async");
    }

    __host__ void cudaMemcpyD2H(std::vector<descriptor_t>& h_result, int sizeOfResult)
    {
        h_result.resize(sizeOfResult);
        cudaStatus = cudaMemcpyAsync(h_result.data(), d_result, sizeof(descriptor_t) * sizeOfResult, cudaMemcpyDeviceToHost, stream);
        assert(cudaStatus == cudaSuccess && "not able to tansfer Data!");
    }

    __host__ void match_gpu_caller(const cudaStream_t& providedStream, int queryCount, int trainCount);

    __host__ void Kernel(int kpSize);
    
    __host__ void getSmallElements(std::vector<float2>& arr, std::vector<float2>& output, float exclude, cudaStream_t& providedStream);

    __host__ void sync(cudaStream_t &providedStream)
    {
        cudaStatus = cudaStreamSynchronize(providedStream);
        assert(cudaStatus == cudaSuccess && "Failed to sync");
    }

    __host__ void cudaFreeAcyncMatcher(cudaStream_t &provided_stream)
    {
        cudaStatus = cudaFreeAsync(d_query,		 provided_stream);
        assert(cudaStatus == cudaSuccess && "cuda free async");
        cudaStatus = cudaFreeAsync(d_train,		 provided_stream);
        assert(cudaStatus == cudaSuccess && "cuda free async2");
        cudaStatus = cudaFreeAsync(d_resMatcher, provided_stream);
        assert(cudaStatus == cudaSuccess && "cuda free async3");
    }
	__host__ void cudaFreeManaged()
	{
		cudaStatus = cudaFree(d_query);
		assert(cudaStatus == cudaSuccess && "cuda free async");
		cudaStatus = cudaFree(d_train);
		assert(cudaStatus == cudaSuccess && "cuda free async2");
		cudaStatus = cudaFree(d_resMatcher);
		assert(cudaStatus == cudaSuccess && "cuda free async3");
	}

    __host__ ~CudaKeypoints()
    {
        cudaFree(d_image);
        cudaFree(d_result);// it was said to -> cudaFree ( void* devPtr )Frees memory on the device.
        cudaFree(d_kp);
        cudaStreamDestroy(stream);
    }
private:
    //-------Provided By Main-------
    const std::vector<uint8_t> &img;
    const int height, width;
    //------Provided By Class,------
    //should be deleted after using
    uint32_t* d_query;
    uint32_t* d_train;
    uint16_t* d_resMatcher;
    uint8_t* d_image;
    uint8_t* d_result;
    float2* d_kp;
    cudaStream_t stream;
    cudaError_t cudaStatus;
};

class Cuda
{
public:
	Cuda() = default;
    Cuda(const int& height, const int& width, cudaError_t cudaStatus)
        :height(height), width(width)
        , d_data(nullptr), d_result(nullptr), cudaStatus(cudaStatus)
    {
        
    }
    
    __host__
        void memoryAllocation(cudaStream_t &providedStream, int size)
    {
        cudaStatus = cudaMallocAsync((void**)&d_data, (size+ 2) * sizeof(uint16_t), providedStream);
		assert(cudaStatus == cudaSuccess && "cudaMalloc failed!");

        cudaStatus = cudaMallocAsync((void**)&d_result, sizeof(uint8_t) * size * 4, providedStream);
        assert(cudaStatus == cudaSuccess && "cudaMalloc failed!");
    }

	__host__
		void standardMemoryAllocation(cudaStream_t& providedStream, int sizeInBytes, int resultSize)
	{
		cudaStatus = cudaMallocAsync((void**)&d_data, sizeInBytes, providedStream);
		assert(cudaStatus == cudaSuccess && "cudaMalloc failed!");

		cudaStatus = cudaMallocAsync((void**)&d_result, resultSize, providedStream);
		assert(cudaStatus == cudaSuccess && "cudaMalloc failed!");
	}

	__host__ 
		void uploadToDevice(cudaStream_t & providedStream,const std::vector<uint16_t> &data)
	{
		if (data.size() == 0) { throw "the vector is empty"; }
		cudaStatus = cudaMemcpyAsync(d_data, data.data(), sizeof(uint16_t) * data.size(), cudaMemcpyHostToDevice, providedStream);
		assert(cudaStatus == cudaSuccess && "not able to trainsfer data, between host and device");
	}

	__host__
		void uploadToDevice(cudaStream_t& providedStream, const std::vector<float>& data, int size)
	{
		cudaStatus = cudaMemcpyAsync(d_data, data.data(), size, cudaMemcpyHostToDevice, providedStream);
		assert(cudaStatus == cudaSuccess && "not able to trainsfer data, between host and device");
	}
	__host__ 
		void download(cudaStream_t & providedStream,std::vector<uint8_t> &h_Data, int size)
	{
		cudaStatus = cudaMemcpyAsync(h_Data.data(), d_result, sizeof(uint8_t) * size, cudaMemcpyDeviceToHost, providedStream);
		assert(cudaStatus == cudaSuccess && "not able to transfer device to host!");
	}
    __host__
    void rawValue(cudaStream_t& providedStream);

    __host__
    void sync(cudaStream_t & providedStream)
    {
        cudaStatus = cudaStreamSynchronize(providedStream);
    }
    void outPutFile(std::vector<uint8_t> & h_cpy)
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
        cudaFree(d_data);// it was said to -> cudaFree ( void* devPtr )Frees memory on the device.
    }

public:
	uint8_t* d_result;
protected:
	int height, width;
    uint16_t* d_data;
	cudaError_t cudaStatus;
};
