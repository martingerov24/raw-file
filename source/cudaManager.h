#pragma once
#include "parameters.h"
#include "cuda_runtime.h"
#include "device/device_manager.h"
#include <inttypes.h>

struct NVProf {
	NVProf(const char* name);
	~NVProf();
};

//so we could use more than one context and update two pictures simultaneously
///@param Device manages cuda contexts
///@param DeviceBuffer manages memory allocation on the device
///@param CUstream cuda driver api stream
class Cuda
{
public:
    Cuda() = delete;
    Cuda& operator=(Cuda &other) = delete;
    Cuda& operator=(Cuda &&other) = delete;

    Cuda(ImageParams& _params);

    __host__ ~Cuda();

    __host__ void initDeviceAndLoadKernel(const char* kernelPath, const char* kernelFunction);

	__host__ void memoryAllocation(const size_t sizeInBytes, const size_t resultSize);
    
    __host__ void deallocate();
    
    __host__ int upload(void* host);

    __host__ int uploadAsync(void* host);
    
    __host__ void makeCurrent();
    
    __host__ void synchronize();
    
    __host__ int download(void* host);

    __host__ int downloadAsync(void* host);

    __host__ int rawValue();

    void debugOutPutFile(uint8_t*& h_cpy);
protected:
    const ImageParams& params;
    Device& device;
    DeviceBuffer input_buffer;
    DeviceBuffer output_buffer;
    CUstream stream;
};
