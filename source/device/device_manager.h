#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H
#include <cuda.h>
#include <string>
#include <vector>
#include <mutex>

//Parameters
struct Params {
    std::string name;         //< Name of the device as returned by the CUDA API
    std::string pciBusId;     //< Identifier of the PCI lane on which this device resides
    size_t memory;            //< Total memory in bytes
    size_t sharedMemPerBlock; //< Total shared memory per block in bytes
    int ccmajor;              //< CUDA Compute Capability major version number
    int ccminor;              //< CUDA Compute Capability minor version number
    int warpSize;             //< Number of threads in a warp. Most probably 32
    int multiProcessorCount;  //< Number of SMs in the GPU
    int maxThreadsPerBlock;   //< Maximum number of threads that can work concurrently in a block
    int maxThreadsPerMP;      //< Maximum number of threads that can work concurrently in a SM
    int devId;                //< The index of this device
    int busId;                //< Index of the PCI bus on which the device is mounted
    int tccMode;              //< True if device is running in Tesla Compute Cluster mode
    int clockRate;            //< Device clock rate in MHz
    unsigned nvLink;          //< A bitmask, the position of every high bit marks the device index from which we can read memory

    Params() :
        name("Unknown"),
        pciBusId(""),
        memory(0),
        sharedMemPerBlock(0),
        ccmajor(-1),
        ccminor(-1),
        warpSize(-1),
        multiProcessorCount(-1),
        maxThreadsPerBlock(-1),
        maxThreadsPerMP(-1),
        devId(-1),
        busId(-1),
        tccMode(-1),
        clockRate(-1),
        nvLink(0)
	{}
};
struct Device;
struct ThreadData;
// A structure managing host-to-device buffer transfers
// Buffers are just a means to transfer data back and forth device and host
// The buffer is in no way responsible for managing what goes where.
// The programmer is responsible for managing device contexts and setting the right
// context on the CUDA stack prior to every buffer invocation
struct DeviceBuffer {
public:
    friend struct Device;
    friend struct ThreadData;

    DeviceBuffer(const DeviceBuffer& buffer) {}
    void  operator=(const DeviceBuffer& buffer) {}
    DeviceBuffer()
    :	name(""),
        buffer(NULL),
        size(0)
    {
        //blank
    }
    DeviceBuffer(std::string name):
        name(name),
        buffer(NULL),
        size(0)
    {
        //blank
    }
    ~DeviceBuffer() { freeMem(); }
    void init(const std::string& name);
    // Allocate a device buffer of given size
    int alloc(size_t size);
    // Free any resources owned
    int freeMem();
    // Synchronously upload a given host buffer to the device
    int upload(void* host, size_t size);
    // Upload a given host buffer to the device ASYNCHRONOUSLY
    int uploadAsync(void* host, size_t size, CUstream stream);
    //Download a device buffer into the given host pointer. Pointer MUST point to a large enough buffer!
    int download(void* host);
    // Download buffer from the device ASYNCHRONOUSLY
    int downloadAsync(void* host, CUstream stream);
    // Returns the device pointer associated with this DeviceBuffer
    const void* get() const { return buffer; }
    // Returns the device pointer associated with this DeviceBuffer
    void* getNonConst() const { return buffer; }
    // Returns the size of the buffer allocated on the device
    size_t getSize() const { return size; }
private:
    std::string name; // Name of this buffer.
    void* buffer = nullptr; // Pointer to the buffer on the device
    size_t size = 0; // Size of the buffer in bytes
};

struct CompileOptions {
    int maxThreads;
};

// A containter for CUDA device information
struct Device {
    friend struct DeviceManager;
    //Methods
    Device(const Device& device) = delete;
    Device operator=(const Device& device) = delete;
    Device(Device&& device) noexcept = default;
    Device& operator=(Device&& device) noexcept = default;
    //Constructs the device and inits all params with nullptr
    Device():
        handle(-1),
        context(nullptr),
        program(nullptr),
        function(nullptr),
        maxThreads(1024)
    {}
    //freeing memory of context and program
    void freeMem();
    //freemem
    ~Device() {
        freeMem();
    }
    //geting information from the gpu, can be print
    const std::string getInfo() const; // Produce a string out of the contained device info and return it
    // Get the device context
    CUcontext getContext() const {
        return context;
    }
    //get the program associated with the device
    CUmodule getProgram() const {
        return program;
    }
    //get the function param, can be nullptr
    CUfunction& getFunction() {
        return function;
    }
    //get handle of CUdevice as const
    const CUdevice& getHandle() const {
        return handle;
    }
    //handle of CUdevice as non const
    CUdevice& getHandle() {
        return handle;
    }
    //make the context current, or create a new one
    void makeCurrent();
    //set CUfunction
    ///@param filename is the ptx associated with the function you want to run
    ///@param funcitonName is the function to invoke form the file
    CUresult setFunction(const char* filename, const char* functionName);
    //returns the max threads that can be ran on device
    int getMaxThreads() const { return maxThreads; }
private:
    Params params;
    CUdevice handle;   //< Handle to the CUDA device
    CUcontext context = nullptr; //< Handle to the CUDA context associated with this device
    CUmodule program;  //< Handle to the compiled program
    int maxThreads; //< Maximum number of threads allowed per block
    CUfunction function;
};

enum class DeviceError {
    Success,
    NoDevicesFound,
    PtxSourceNotFound,
    InvalidPtx,
};

// Main GPU management class. A singleton class. You cannot create it explicitly
// In order to obtain an instance of it you need to call the getInstance() method
// It is responsible for initializing, deinitializing the GPU devices.
// Can be asked to provide information for a particular device
// The process method would start the processing on the GPU
struct DeviceManager {
    // The only way to obtain an instance of an object is through this method
    static DeviceManager& getInstance(int emulation=0) {
        static DeviceManager instance;
        instance.init(emulation);
        return instance;
    }
    // Frees all resources held by the instance
    int deinit();
    // Destructor. Should call deinit();
    ~DeviceManager();
    // Return a const reference to the device with given index. Caller MUST ensure that the index is less than numDevices!
    Device& getDevice(int index);
    int getDeviceCount() const { return numDevices; }

private:
    // Ask for information about a particular device
    // @param deviceIndex The index of the device that we are querying
    // @param devInfo A container where the device data will be populated
    // @returns 0 on success
    int getDeviceInfo(int deviceIndex, Device &devInfo);
    // Queries the system for GPU devices, sets the numDevices member accordingly, populates the devices list
    // @returns 0 on success
    int init(int emulation);
    // Check whether the GPU manager has successfully been initialized.
    // returns 1 if the init() method has already been called
    int isInitialized() const { return initialized; }
    int initialized;                 //< A flag to tell us whether the class methods are safe to be called.
    int numDevices;                  //< The number of devices in the system. Populated by init()
    std::vector<Device> devices;     //< An array containing per-device information
    DeviceManager();                               //< Private constructor. We don't want anyone to create objects of this type.
    DeviceManager(DeviceManager const&) = delete;  //< Remove copy constructor. We don't want anyone to copy objects of this type.
    void operator=(DeviceManager const&) = delete; //< Remove operator=. We don't want anyone to copy objects of this type.
    std::mutex mtx;
};

#endif // DEVICE_MANAGER_H
