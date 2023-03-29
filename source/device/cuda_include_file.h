#ifndef CUDA_INCLUDE_FILE_H
#define CUDA_INCLUDE_FILE_H

#include <cuda.h>
typedef CUresult GPUResult;
#define GPU_SUCCESS CUDA_SUCCESS

void printMessage(int error, const char* file, int line, const char* func, const char* message);

#define checkError(err)                                           \
if ((err) != CUDA_SUCCESS) {                                      \
    printMessage(int(err), __FILE__, __LINE__, __FUNCTION__, ""); \
    return (err);                                                 \
}

#define checkErrorM(err, msg)                                      \
if ((err) != CUDA_SUCCESS) {                                       \
    printMessage(int(err), __FILE__, __LINE__, __FUNCTION__, msg); \
    return (err);                                                  \
}

#define checkErrorNoRet(err)                                      \
if ((err) != CUDA_SUCCESS) {                                      \
    printMessage(int(err), __FILE__, __LINE__, __FUNCTION__, ""); \
    return;                                                       \
}

#define checkErrorMNoRet(err, msg)                                \
if ((err) != CUDA_SUCCESS) {                                      \
    printMessage(int(err), __FILE__, __LINE__, __FUNCTION__, msg);\
    return;                                                       \
}

// Method is called when a cuda error is encountered to print information
// error code, in what file, what line and which funtion was executed at the time of the error

#endif // CUDA_INCLUDE_FILE_H
