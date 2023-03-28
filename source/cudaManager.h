#pragma once
#include "cuda_runtime.h"
#include "cuda/std/cmath"
#include <inttypes.h>

struct NVProf {
	NVProf(const char* name);
	~NVProf();
};

struct ImageParams {
	ImageParams(
		const int32_t _height,
		const int32_t _width,
		const int32_t _stride,
        const int32_t _bpp
	);

	inline size_t size() const { 
        return (height * stride); 
    }

    inline size_t numberOfPixels() const {
        return (height * width);
    }

	const int32_t height;
	const int32_t width;
	const int32_t stride;
    const int32_t bpp;
	const int8_t channels = 4;
};

class Cuda
{
public:
    Cuda(
        ImageParams& _params,
        cudaError_t _cudaStatus
    );

    __host__ ~Cuda();
    
	__host__ void memoryAllocation(cudaStream_t& providedStream, const size_t sizeInBytes, const size_t resultSize);

    __host__ void deallocate();

	__host__ void uploadToDevice(cudaStream_t& providedStream, const uint8_t* data);

    __host__ void download(cudaStream_t & providedStream, uint8_t*& h_Data);

    __host__ void rawValue(cudaStream_t& providedStream);

    __host__ void sync(cudaStream_t & providedStream);

    void debugOutPutFile(uint8_t*& h_cpy);
protected:
    uint8_t* d_data;
	uint8_t* d_result;
    const ImageParams& params;

    size_t m_sizeInBytes = 0; 
    size_t m_resultSize = 0;
	cudaError_t cudaStatus;
};
