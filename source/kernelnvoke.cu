#include "cudaManager.h"

__device__ __forceinline__
uint8_t readColorFromUnpackedRawData(uint16_t number) {
	// this is the solution if little indian
	uint8_t first_8_bits = number & 0b11111111; first_8_bits = first_8_bits >> 4;
	number = number >> 8;
	uint8_t n = number & 0b11111111; n = n >> 4;
	n = n & 0b1111; // basicly the paddings are throun away
	// now we have 2 4 bit numbers and when combining them OR || XOR
	n = n << 4;
	n |= first_8_bits;
	// the second number is our putput
	return n;
}

__device__
void kernelForRawInput(
	uint16_t* __restrict__ d_Data
	, uint32_t* __restrict__ cpy_Data
	, const int width
	, const int height 
) {
	short x = (blockIdx.x * blockDim.x) + threadIdx.x;
	short y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= width || 
		y >= height || 
		x < 0 ||
		y < 0
	) {
		return;
	}

	int calc = y * width + x;
	uint8_t n = d_Data[calc] >> 2;//readColorFromUnpackedRawData(d_Data[calc]);

	uint8_t idx = (y & 1) + !(x & 1);
	uint8_t rgb[4] = { 0, 0, 0, 1 };
	rgb[idx] = n;

	cpy_Data[calc] = *reinterpret_cast<int*>(rgb);
}
