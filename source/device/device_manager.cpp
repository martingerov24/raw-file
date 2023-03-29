#include "device_manager.h"
#include "cuda_include_file.h"
#include <cassert>
#include <fstream>

DeviceManager::DeviceManager() : initialized(0), numDevices(0) {}
DeviceManager::~DeviceManager() { deinit(); }

int DeviceManager::init(int emulation) {
    CUresult err = GPU_SUCCESS;
    //Nothing to do if already initialized.
    if (initialized) {
        return err;
    }
    std::lock_guard<std::mutex> lck (mtx);
    if (initialized) {
        return err;
    }

    err = cuInit(0);
    checkError(err);

    err = cuDeviceGetCount(&numDevices);
    checkError(err);

    devices.reserve(numDevices);
    for (int i = 0; i < numDevices; i++) {
        devices.emplace_back(std::move(Device()));
        Device& devInfo = devices[i];
        int err = getDeviceInfo(i, devInfo);
        checkError(err);
    }
    initialized = 1;
    return err;
}

int DeviceManager::getDeviceInfo(int deviceIndex, Device &devInfo) {
    CUresult err = GPU_SUCCESS;

    devInfo.params.devId = deviceIndex;
    CUdevice device = 0;
    err = cuDeviceGet(&device, deviceIndex);
    checkError(err);
    devInfo.handle = device;

    char pciBusId[256];
    err = cuDeviceGetPCIBusId(pciBusId, 256, device);
    checkError(err);
    devInfo.params.pciBusId = std::string(pciBusId);

    char name[256];
    err = cuDeviceGetName(name, 256, device);
    checkError(err);
    devInfo.params.name = std::string(name);

    CUdevprop prop;
    err = cuDeviceGetProperties(&prop, device);
    checkError(err);

    devInfo.params.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    devInfo.params.sharedMemPerBlock = prop.sharedMemPerBlock;
    devInfo.params.warpSize = prop.SIMDWidth;
    devInfo.params.clockRate = prop.clockRate;

    int value;
    CUdevice_attribute attrib;

    attrib = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
    err = cuDeviceGetAttribute(&value, attrib, device);
    checkError(err);
    devInfo.params.ccmajor = value;

    attrib = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
    err = cuDeviceGetAttribute(&value, attrib, device);
    checkError(err);
    devInfo.params.ccminor = value;

    attrib = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
    err = cuDeviceGetAttribute(&value, attrib, device);
    checkError(err);
    devInfo.params.maxThreadsPerMP = value;

    attrib = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
    err = cuDeviceGetAttribute(&value, attrib, device);
    checkError(err);
    devInfo.params.multiProcessorCount = value;

    attrib = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
    err = cuDeviceGetAttribute(&value, attrib, device);
    checkError(err);
    devInfo.params.busId = value;

    attrib = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TCC_DRIVER;
    err = cuDeviceGetAttribute(&value, attrib, device);
    checkError(err);
    devInfo.params.tccMode = value;

    size_t bytes;
    err = cuDeviceTotalMem(&bytes, device);
    checkError(err);
    devInfo.params.memory = bytes;

    CUcontext ctx = nullptr;
    int flags = CU_CTX_SCHED_AUTO;
    err = cuCtxCreate(&ctx, flags, device);
    checkError(err);
    devInfo.context = ctx;

    return err != GPU_SUCCESS;
}

Device& DeviceManager::getDevice(int index) {
    return devices[index];
}

int DeviceManager::deinit() {
    devices.clear();
    initialized = 0;
    numDevices = 0;
    CUresult err = GPU_SUCCESS;
    checkError(err);
    return err;
}

/////////////////////////////////////////////////////////////////////////////////

void DeviceBuffer::init(const std::string& name) {
    this->name = name;
}

int DeviceBuffer::alloc(size_t size) {
    CUresult err = GPU_SUCCESS;
    if (buffer) {
        freeMem();
    }
    if (size != 0) {
        err = cuMemAlloc((CUdeviceptr*)&buffer, size);
        assert(err == CUDA_SUCCESS);
    }
    this->size = size;
    return err != GPU_SUCCESS;
}

int DeviceBuffer::upload(void* host, size_t size) {
    if (!buffer) return CUDA_ERROR_NOT_INITIALIZED;
    if (size > this->size) return CUDA_ERROR_OUT_OF_MEMORY;

    CUresult err = GPU_SUCCESS;
    err = cuMemcpyHtoD((CUdeviceptr)buffer, host, size);
    return err != GPU_SUCCESS;
}

int DeviceBuffer::uploadAsync(void* host, size_t size, CUstream stream) {
    if (!buffer) return CUDA_ERROR_NOT_INITIALIZED;
    if (size > this->size) return CUDA_ERROR_OUT_OF_MEMORY;

    CUresult err = GPU_SUCCESS;
    err = cuMemcpyHtoDAsync((CUdeviceptr)buffer, host, size, stream);
    assert(err == CUDA_SUCCESS && "cuMemcpyHtoDAsync");
    return err != GPU_SUCCESS;
}

int DeviceBuffer::download(void* host) {
    assert(host != nullptr);
    CUresult err = cuMemcpyDtoH(host, (CUdeviceptr)buffer, size);
    checkError(err);
    return err != GPU_SUCCESS;
}

int DeviceBuffer::downloadAsync(void* host, CUstream stream) {
    assert(host != nullptr);
    CUresult err = cuMemcpyDtoHAsync(host, (CUdeviceptr)buffer, size, stream);
    checkError(err);
    return err != GPU_SUCCESS;
}

int DeviceBuffer::freeMem() {
    CUresult err = GPU_SUCCESS;
    if (buffer) {
        err = cuMemFree((CUdeviceptr)buffer);
        assert(err == CUDA_SUCCESS);
        buffer = NULL;
        size = 0;
    }
    return err != GPU_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////

void Device::makeCurrent() {
    CUcontext currentCtx = nullptr;
    if (context == nullptr) {
        int flags = CU_CTX_SCHED_AUTO;
        CUresult err = cuCtxCreate(&context, flags, handle);
        assert(err == CUDA_SUCCESS);
    }

    CUresult err = cuCtxGetCurrent(&currentCtx);
    assert(err == CUDA_SUCCESS);

    if (currentCtx == context) {
        return;
    }

    err = cuCtxSetCurrent(context);
    assert(err == CUDA_SUCCESS);
}

bool fileExists(const std::string& filename) {
	std::fstream file(filename);
	if (!file.is_open()) {
		return false;
	}
	return true;
}

CUresult Device::setFunction(
    const char* filename,
    const char* functionName
){
    if (context == nullptr) {
        printf("create a contex and set it current first\n");
		return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    }
    if(filename == nullptr) {
        printf("ptx filename is nullptr\n");
		return CUDA_ERROR_INVALID_PTX;
    }
    if (!fileExists(filename)) {
        printf("ptx file %s does not exist\n", filename);
		return CUDA_ERROR_INVALID_PTX;
    }
    CUmodule module = nullptr;
    CUresult cudaStatus = cuModuleLoad(&module, filename);
    checkErrorM(cudaStatus, "could not load module");

    cudaStatus = cuModuleGetFunction(&getFunction(), module, functionName);
    checkErrorM(cudaStatus, "could not get funciton");

	return cudaStatus;
}

const std::string Device::getInfo() const {
    std::string message = "";

    const float totalMemory = float(params.memory) / float(1024 * 1024 * 1024);
    const float sharedMemory = float(params.sharedMemPerBlock) / float(1024);

    std::string driverMode = "";
    switch(params.tccMode) {
        case 0: driverMode = "WDDM"; break;
        case 1: driverMode = "TCC"; break;
        default:
            driverMode = "Unknown"; break;
    }

    const int buffSize = 1024;
    char buff[buffSize];
    snprintf(buff, buffSize, "Device[%d] is : %s\n\n", params.devId, params.name.c_str()); message += buff;
    snprintf(buff, buffSize, "\tPCI Bus ID                     : %s\n", params.pciBusId.c_str());  message += buff;
    snprintf(buff, buffSize, "\tDriver mode                    : %s\n", driverMode.c_str());  message += buff;
    snprintf(buff, buffSize, "\tClock Rate                     : %dMhz\n", params.clockRate/1000);   message += buff;
    snprintf(buff, buffSize, "\tTotal global memory            : %.1f GB\n", totalMemory);    message += buff;
    snprintf(buff, buffSize, "\tShared memory                  : %.1f KB\n", sharedMemory);   message += buff;
    snprintf(buff, buffSize, "\tCompute Capability             : %d.%d\n", params.ccmajor, params.ccminor);     message += buff;
    snprintf(buff, buffSize, "\tWarp size                      : %d\n", params.warpSize);            message += buff;
    snprintf(buff, buffSize, "\tMax threads per block          : %d\n", params.maxThreadsPerBlock);  message += buff;
    snprintf(buff, buffSize, "\tMax threads per multiprocessor : %d\n", params.maxThreadsPerMP);     message += buff;
    snprintf(buff, buffSize, "\tNumber of multiprocessors      : %d\n", params.multiProcessorCount); message += buff;

    message += "\n";
    return message;
}

void Device::freeMem() {
    if (program) {
        CUresult res = cuModuleUnload(program);
        assert(res == CUDA_SUCCESS);
        program = nullptr;
    }
    if (context) {
        CUresult res = cuCtxDestroy(context);
        assert(res == CUDA_SUCCESS);
        context = nullptr;
    }
}
