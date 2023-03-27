#include "cudaManager.h"
#include "nvtx3/nvToolsExt.h"

#include <chrono>
#include <cassert>

NVProf::NVProf(const char* name) {
    nvtxRangePush(name);
}

NVProf::~NVProf() {
    nvtxRangePop();
}

ImageParams::ImageParams(
    const int32_t _height,
    const int32_t _width,
    const int32_t _stride,
    const int32_t _bpp
) : height(_height)
, width(_width)
, stride(_stride)
, bpp(_bpp) {}

Cuda::Cuda(
    ImageParams& _params,
    cudaError_t _cudaStatus
) :params(_params)
, d_data(nullptr)
, d_result(nullptr)
, cudaStatus(_cudaStatus)
{}

__host__
void Cuda::memoryAllocation(cudaStream_t& providedStream, const size_t sizeInBytes, const size_t resultSize) {
    m_sizeInBytes = sizeInBytes;
    m_resultSize = resultSize;

    cudaStatus = cudaMallocAsync((void**)&d_data, m_sizeInBytes, providedStream);
    assert(cudaStatus == cudaSuccess && "cudaMalloc failed!");

    cudaStatus = cudaMallocAsync((void**)&d_result, m_resultSize, providedStream);
    assert(cudaStatus == cudaSuccess && "cudaMalloc failed!");
}

__host__
void Cuda::deallocate() {
    cudaStatus = cudaFree(d_data);// it was said to -> cudaFree ( void* devPtr )Frees memory on the device.
    assert(cudaStatus == cudaSuccess && "not able to deallocate d_data");
    cudaStatus = cudaFree(d_result);
    assert(cudaStatus == cudaSuccess && "not able to deallocate d_result");
}

__host__
void Cuda::uploadToDevice(cudaStream_t& providedStream, const uint8_t* data) {
    // If m_sizeInBytes is 0 we have not allocated enought memory.
    assert(m_sizeInBytes != 0);
    cudaStatus = cudaMemcpyAsync(d_data, data, m_sizeInBytes, cudaMemcpyHostToDevice, providedStream);
    assert(cudaStatus == cudaSuccess && "not able to trainsfer data, between host and device");
}

__host__ 
void Cuda::download(cudaStream_t & providedStream, uint8_t*& h_Data) {
    // If m_resultSize is 0, we have not allocated enought memory 
    assert(m_resultSize != 0);
    cudaStatus = cudaMemcpyAsync(h_Data, d_result, m_resultSize, cudaMemcpyDeviceToHost, providedStream);
    assert(cudaStatus == cudaSuccess && "not able to transfer device to host!");
}

__host__
void Cuda::sync(cudaStream_t & providedStream) {
    cudaStatus = cudaStreamSynchronize(providedStream);
    assert(cudaStatus == cudaSuccess && "not able to sync!");
}

void Cuda::debugOutPutFile(uint8_t*& h_cpy) {
    FILE* fileWr;
    fileWr = fopen("writingFile.ppm", "w+");
    fprintf(fileWr, "%s %d %d %d ", "P6", params.width , params.height, 255);
    fclose(fileWr);

    fileWr = fopen("writingFile.ppm", "ab+");
    fwrite(reinterpret_cast<const char*>(&h_cpy[0]), 1, params.size(), fileWr);
    fclose(fileWr);
    fileWr = nullptr;
}

__host__
Cuda::~Cuda() {
    deallocate();
}