#define PROFILING 1
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stbi_write.h"

#include "cudaManager.h"
#include "windowManager.h"

#include <string>
#include <vector>

#if PROFILING 
	#include "minitrace.h"
#endif /*PROFILING*/


bool loadJpgImage(std::vector<uint8_t> &image, int &width, int &height, int& channels) {
	unsigned char* img = stbi_load("image.jpg", &width, &height, &channels, 1);
	if (img == NULL) { return false; }
	image.resize(width * height);
	memcpy(&image[0], img, image.size());
	if (image.empty()){	return false; }
	stbi_image_free(img);
	return true;
}

std::vector<uint8_t> readingRawFile(const char* fileName, ImageParams& params) {
	FILE* rdFile = fopen(fileName, "rb+");
	std::vector<uint8_t> data;
	if (rdFile == NULL) {
		printf("no file found!");
		return data;
	}
	//todo check the size we read == file size
	data.resize(params.size());
	fread(reinterpret_cast<char*>(&data[0]), 1, params.size(), rdFile);
	fclose(rdFile);
	return data;
}

struct CPUProf {
	CPUProf(std::string& thread, std::string&name)
		:thread(thread), name(name) {
		MTR_BEGIN(thread, name);
	}
	~CPUProf() {
		MTR_END(thread, name);
	}
	std::string& thread;
	std::string& name;
};

#define CPUPROF_SCOPE(x, y) CPUProf __cpuprof(x,y);

void processWithCudaAndSaveJpg(const std::vector<uint8_t>& data, ImageParams& params) {
	std::vector<uint8_t> h_result(params.numberOfPixels() * params.channels);

	Cuda cuda(params);
	cuda.initDeviceAndLoadKernel("kernelnvoke.ptx", "kernelForRawInput");
	cuda.memoryAllocation(data.size(), h_result.size());
	cuda.uploadAsync((void*)data.data());
	
	cuda.rawValue();
	cuda.downloadAsync(h_result.data());
	cuda.synchronize();

	stbi_write_jpg("rawFileViewer.jpg", params.width, params.height, params.channels, h_result.data(), params.width * sizeof(int));
}

void processAndDisplayRawImage(const std::vector<uint8_t>& data, ImageParams& params) {
	// OpenGL
	WindowManager window;
	window.init();
	// Data input and output parameters
	std::vector<uint8_t> h_result(params.numberOfPixels() * params.channels);
	// Cuda manager creation
	cudaStream_t stream = nullptr;
	Cuda cuda(params);
	
	cuda.initDeviceAndLoadKernel("kernelnvoke.ptx", "kernelForRawInput");
	cuda.memoryAllocation(data.size(), h_result.size());
	cuda.uploadAsync((void*)data.data());
	
	cuda.rawValue();
	cuda.downloadAsync(h_result.data());
	cuda.synchronize();
	
	while (window.shouldClose()) {
		cuda.rawValue();
		cuda.download(h_result.data());
		cuda.synchronize();
		if(window.draw(h_result.data(), params) == false) {
			break;
		}
	}
	window.terminate();
}

std::vector<uint8_t> processPackedData(const std::vector<uint8_t>& dataByte, ImageParams& params) {
	assert(dataByte.size() == params.size());
	assert(dataByte.size() != params.numberOfPixels());
	
	std::vector<uint16_t> result;
	result.reserve(params.numberOfPixels());

	Cast64toPacked64 first_64_bits;
	Cast16toPacked16 last_16_bits;

	for(int i = 0; i < params.height; i++) {
		const uint8_t* line = dataByte.data() + i * params.stride;
		int numProcessed = 0;
		while (numProcessed < params.width) {
			first_64_bits.element =
				uint64_t(line[0]) << 56 | 
				uint64_t(line[1]) << 48 | 
				uint64_t(line[2]) << 40 | 
				uint64_t(line[3]) << 32 | 
				uint64_t(line[4]) << 24 | 
				uint64_t(line[5]) << 16 | 
				uint64_t(line[6]) << 8  | 
				uint64_t(line[7]);
			last_16_bits.element = uint16_t(line[8]) << 8 | line[9];

			result.emplace_back(uint16_t(first_64_bits.casted.A));
			result.emplace_back(uint16_t(first_64_bits.casted.B));
			result.emplace_back(uint16_t(first_64_bits.casted.C));
			result.emplace_back(uint16_t(first_64_bits.casted.D));
			result.emplace_back(uint16_t(first_64_bits.casted.E));
			result.emplace_back(uint16_t(first_64_bits.casted.F));

			uint16_t first = uint16_t(first_64_bits.casted.remainder);
			uint16_t second = uint16_t(last_16_bits.casted.remainder);

			result.emplace_back(uint16_t(first << 6 | second));
			result.emplace_back(uint16_t(last_16_bits.casted.B));
			line+=10;
			numProcessed +=8;
		}
	}
	
	uint8_t* resDataInBytes = reinterpret_cast<uint8_t*>(result.data());
	std::vector<uint8_t> res(params.numberOfPixels()*sizeof(uint16_t));
	memcpy(&res[0], result.data(), params.numberOfPixels()*sizeof(uint16_t));
	return res;
}

int main() {
	unsigned int saveJpg = 0;
	unsigned int i = 1;
    char *c = (char*)&i;
    if (*c == 0) { //working with only Little endian
		return -1;
	}

	char* fileName = "../rawFile.raw";
	ImageParams params(3000, 4000, 5008, 10);
	std::vector<uint8_t> data = readingRawFile(fileName, params);
	if (params.bpp != 16) {
		std::vector<uint8_t> res = processPackedData(data, params);
		data.swap(res);
	}
	if(saveJpg) {
		processWithCudaAndSaveJpg(data,params);
	} else {
		processAndDisplayRawImage(data, params);
	}
	return 0;
}
