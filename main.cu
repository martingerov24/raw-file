#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "cudaManager.h"

#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define PROFILING 1
#if PROFILING 
	#include "../minitrace/minitrace.h"
	#include "../minitrace/minitrace.c"
#endif 
// PROFILING

bool loadJpgImage(std::vector<uint8_t> &image, int &width, int &height, int& channels) {
	unsigned char* img = stbi_load("image.jpg", &width, &height, &channels, 1);
	if (img == NULL) { return false; }
	image.resize(width * height);
	memcpy(&image[0], img, image.size());
	if (image.empty()){	return false; }
	stbi_image_free(img);
	return true;
}

std::vector<uint8_t> readingRawFile(const char* fileName, ImageParams& params);

void bindTexture(GLuint texture) {
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
}

void onNewFrame() {
	glfwPollEvents();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void createContext(GLFWwindow* &window) {
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
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

#define NVPROF_SCOPE(X) NVProf __nvprof(X);
#define CPUPROF_SCOPE(x, y) CPUProf __cpuprof(x,y);

void processAndDisplayRawImage(const std::vector<uint8_t>& data, ImageParams& params) {
	cudaError_t cudaStatus = cudaError_t(0);
	cudaStream_t stream;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaStatus = cudaSetDevice(0);
	assert(cudaStatus == cudaSuccess && "you do not have cuda capable device!");
	cudaStatus = cudaStreamCreate(&stream);
	
	const size_t inputSize = data.size();
	const size_t resultSize = params.numberOfPixels() * 4;
	std::vector<uint8_t> h_result(resultSize);
	const uint8_t* input = data.data();
	uint8_t* output = h_result.data();
	
	Cuda cuda(params, cudaStatus);
	cuda.memoryAllocation(stream, inputSize, resultSize);
	cuda.uploadToDevice(stream, input);
	//TEST run
	cuda.rawValue(stream);
	cuda.download(stream, output);

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	if (!glfwInit()) {
		throw "glfwInit() FAILED!";
	}

	GLFWwindow* window = glfwCreateWindow(800, 600, "Raw-File Viewer", NULL, NULL);

	if (!window) {
		glfwTerminate();
		throw "no window created";
	}
	cuda.sync(stream);
	createContext(window);

	bool is_show = true;
	GLuint texture;
	glGenTextures(1, &texture);

	while (!glfwWindowShouldClose(window)) 
	{
		{
			NVPROF_SCOPE("raw - value and downloading");
			cuda.rawValue(stream);
			cuda.download(stream, output);
		}
		onNewFrame();
		ImGui::Begin("raw Image", &is_show);
		bindTexture(texture);
		cuda.sync(stream);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, params.width, params.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, output);
		ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)), ImVec2(800, 600));
		ImGui::End();
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glDeleteTextures(sizeof(texture), &texture);
		glfwSwapBuffers(window);
	}

	ImGui_ImplGlfw_Shutdown();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
	glfwDestroyWindow(window);
}

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

std::vector<uint8_t> processPackedData(const std::vector<uint8_t>& dataByte, ImageParams& params) {
	assert(dataByte.size() == params.size());
	assert(dataByte.size() != params.numberOfPixels());
	
	std::vector<uint16_t> result;
	result.reserve(params.numberOfPixels());
	//const uint16_t* data = reinterpret_cast<const uint16_t*>(dataByte.data());

	Cast64toPacked64 first_64_bits;
	Cast16toPacked16 last_16_bits;

	// for(int i = 0; i < params.height/2; i++) {
	// 	for(int j = 0; j < params.width; j+=5) {
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
	//return std::vector<uint8_t>(&resDataInBytes[0], &resDataInBytes[params.numberOfPixels()*2]); //result
}

int main() {
	unsigned int i = 1;
    char *c = (char*)&i;
    if (*c == 0) { //working with only Little endian
		return -1;
	}

	char* fileName = "../../rawFile.raw";
	ImageParams params(3000, 4000, 5008, 10);
	std::vector<uint8_t> data = readingRawFile(fileName, params);
	if (params.bpp != 16) {
		std::vector<uint8_t> res = processPackedData(data, params);
		data.swap(res);
	} 
	processAndDisplayRawImage(data, params);

	return 0;
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
