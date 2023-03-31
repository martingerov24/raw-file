#pragma once
#include <cinttypes>
#include <cstddef>

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

struct Pack64Bytes {
	uint64_t remainder : 4;
	uint64_t F : 10;
	uint64_t E : 10;
	uint64_t D : 10;
	uint64_t C : 10;
	uint64_t B : 10;
	uint64_t A : 10;
};

struct Pack16Bytes {
	uint16_t B : 10;
	uint16_t remainder : 6;
};

union Cast64toPacked64 {
	Pack64Bytes casted;
	uint64_t element;
};

union Cast16toPacked16 {
	Pack16Bytes casted;
	uint16_t element;
};