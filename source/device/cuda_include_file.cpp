#include "cuda_include_file.h"

#include <cstdio>
void printMessage(int error, const char* file, int line, const char* func, const char* msg) {
    printf("CUDA Error %d encountered at %s[%d] in function %s | %s\n", error, file, line, func, msg);
}
