#include "cudaManager.h"
#include "nvtx3/nvToolsExt.h"
#include "cuda_include_file.h"
#include <chrono>
#include <cassert>

#define THREADS_PER_BLOCK 32

NVProf::NVProf(const char* name) {
    nvtxRangePush(name);
}

NVProf::~NVProf() {
    nvtxRangePop();
}
#define NVPROF_SCOPE(X) NVProf __nvprof(X);

Device& getDevice(int device = 0) {
    DeviceManager &devman = DeviceManager::getInstance();
    if (devman.getDeviceCount() == 0) {
        throw std::runtime_error("No cuda devices");
    }
    return devman.getDevice(device);
}

CUstream& getStream(){
    static CUstream stream;
    static bool initialized = false;
    if (initialized == false) {
        return stream;
    }
    CUresult err = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    if (err == CUDA_SUCCESS) {
        initialized = true;
    } else {
        printf("Error initializing cuda stream\n");
    }
    return stream;
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
    ImageParams& _params
): params(_params)
, input_buffer("InputBuffer")
, output_buffer("OutputBuffer")
, device(getDevice()) 
{ }

__host__
void Cuda::initDeviceAndLoadKernel(const char* kernelPath, const char* kernelFunction) {
    makeCurrent();
    CUresult err = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    assert(err == CUDA_SUCCESS);

    CompileOptions opts;
    opts.maxThreads = THREADS_PER_BLOCK;
    err = device.setSource(kernelPath, opts);
    err = device.setFunction(kernelPath, kernelFunction);
    checkErrorNoRet(err);
    synchronize();
}

__host__
void Cuda::memoryAllocation(const size_t sizeInBytes, const size_t resultSize) {
    input_buffer.init(std::string("InputBuffer"));
    CUresult err = CUresult(input_buffer.alloc(sizeInBytes));
    checkErrorMNoRet(err, "could not allocate buffer");

    output_buffer.init(std::string("OutputBuffer"));
    err = CUresult(output_buffer.alloc(resultSize));
    checkErrorMNoRet(err, "could not allocate buffer");
}

__host__
void Cuda::deallocate() {
    if (input_buffer.get() != nullptr) {
        input_buffer.freeMem();
    } 
    if (output_buffer.get() != nullptr) {
        output_buffer.freeMem();
    }
}

void Cuda::makeCurrent() {
    device.makeCurrent();
}

__host__ 
int Cuda::upload(void* host) {
    return input_buffer.upload(host, input_buffer.getSize());
}

__host__ 
int Cuda::uploadAsync(void* host) {
    makeCurrent();
    return input_buffer.uploadAsync(
        host,
        input_buffer.getSize(),
        stream
    );
}

__host__
int Cuda::rawValue() {
    const int blockDimX = ((params.width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    const int blockDimY = ((params.height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    const void* input = input_buffer.get();
    const void* output = output_buffer.get(); 
    void* args[] = { 
        (void*)&input,
        (void*)&output,
        (void*)&params.width, 
        (void*)&params.height 
    };
    CUresult cudaStatus = cuLaunchKernel(device.getFunction(), blockDimX, blockDimY, 1, 32, 32, 1, 0, stream, args, nullptr);
    checkError(cudaStatus);
    return cudaStatus;
}

__host__
int Cuda::download(void* host) {
    return output_buffer.download(host);
}

__host__ 
int Cuda::downloadAsync(void* host) {
    return output_buffer.downloadAsync(host, stream);
}

__host__ 
void Cuda::synchronize() {
    CUresult err = cuStreamSynchronize(stream);
    checkErrorMNoRet(err, "could not synchronize");
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
    device.freeMem();
}