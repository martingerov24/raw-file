#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include <stdio.h>
#include <vector>
#include <string>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../../header/stb_image.h"
//#define STB_IMAGE_RESIZE_IMPLEMENTATION
//#include "build/stb_image_resize.h"
#include "../../header/CudaClass.h"
#include <iostream>
#include <inttypes.h>

#define PROFILING 1
#if PROFILING 
#include "../minitrace/minitrace.h"
#include "../minitrace/minitrace.c"
#endif 
// PROFILING

extern std::vector<uint16_t> matcher_result;

bool load(std::vector<uint8_t> &image,  int &width, int &height, int& channels)
{
	unsigned char* img = stbi_load("image.jpg", &width, &height, &channels, 1);
	//stbir_resize_uint8(img, width, height, 0, img, width, height, 0, 1);
	if (img == NULL) { return false; }
	image.resize(width * height);
	memcpy(&image[0], img, image.size());
	if (image.empty()){	return false; }
	stbi_image_free(img);
	return true;
}

std::vector<uint16_t> ReadingFiles(char* fileName, int height, int width);

void bindTexture(GLuint texture)
{
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
}
void onNewFrame()
{
	glfwPollEvents();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}
void createContext(GLFWwindow* &window)
{
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
}

void Loop(const std::vector<uint16_t>& data, const int height, const int width)
{
	cudaError_t cudaStatus = cudaError_t(0);
	cudaStream_t stream;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaStatus = cudaSetDevice(0);
	assert(cudaStatus == cudaSuccess && "you do not have cuda capable device!");
	cudaStatus = cudaStreamCreate(&stream);

	std::vector<uint8_t> h_result;
	int size = height * width;
	h_result.resize(size*4);
	
	Cuda cuda(height, width, cudaStatus);
	cuda.memoryAllocation(stream, size);
	cuda.uploadToDevice(stream, data);
	cuda.rawValue(stream);
	cuda.download(stream, h_result, size * 4);

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
		cuda.rawValue(stream);
		cuda.download(stream, h_result, size * 4);
		onNewFrame();
		ImGui::Begin("raw Image", &is_show);
		bindTexture(texture);
		cuda.sync(stream);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_result.data());
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

struct NVProf {
	NVProf(const char* name) {
		nvtxRangePush(name);
	}
	~NVProf() {
		nvtxRangePop();
	}
};

struct CPUProf
{
	CPUProf(std::string& thread, std::string&name)
		:thread(thread), name(name)
	{
		MTR_BEGIN(thread, name);
	}
	~CPUProf()
	{
		MTR_END(thread, name);
	}
	std::string& thread;
	std::string& name;
};

#define NVPROF_SCOPE(X) NVProf __nvprof(X);
#define CPUPROF_SCOPE(x, y) CPUProf __cpuprof(x,y);

void MatchKernel_Result(const std::vector<uint8_t>& data, const int height, const int width)
{
#ifdef PROFILING
	mtr_init("../trace.json");
	MTR_META_PROCESS_NAME("MINITRACE_test");
	MTR_META_THREAD_NAME("main thread");

	int long_running_thing_1;
	int long_running_thing_2;
	MTR_START("background", "long_running", &long_running_thing_1);
	MTR_START("background", "long_running", &long_running_thing_2);
#endif // PROFILING
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	std::vector<uint16_t> h_result;
	int size = height * width;
	
	std::vector<descriptor_t> query;
	std::vector<descriptor_t> train;
	CudaKeypoints cuda(data, height, width);

	cuda.startup(size, leftPoint.size() / 2);

	{
		NVPROF_SCOPE("Kernel plus copy from cpu-gpu");
		cuda.cudaUploadKeypoints(leftPoint);
		cuda.Kernel(leftPoint.size() / 2);
		cuda.cudaMemcpyD2H(query, leftPoint.size() / 2);
	}
	cuda.sync(stream);

	{
		NVPROF_SCOPE("Second kernel cpu-gpu");
		cuda.cudaUploadKeypoints(rightPoint);
		cuda.Kernel(rightPoint.size() / 2);
		cuda.cudaMemcpyD2H(train, rightPoint.size() / 2);
	}

	cuda.sync(stream);

	cuda.MemoryAllocationAsync(stream, query.size(), train.size());
	{
		CPUPROF_SCOPE(std::string("main"), std::string("for 1000 iterations -> matcher"));
		for (int i = 0; i < 1000; ++i)
		{
			NVPROF_SCOPE("for a single iteration on match kernel");
			cuda.MemcpyUploadAsyncForMatches(stream, query, train);
			cuda.match_gpu_caller(stream, query.size(), train.size());
			cuda.downloadAsync(stream, h_result, query.size());
			cuda.sync(stream);
		}
		cuda.cudaFreeAcyncMatcher(stream);
	}
	for (int i = 0; i < 100; i++)
	{
		if (h_result[i] != matcher_result[i])
		{
			printf("wrong_answer\n");
		}
	}
	//33.36
#ifdef PROFILING
	MTR_FINISH("background", "long_running", &long_running_thing_1);
	MTR_FINISH("background", "long_running", &long_running_thing_2);
	mtr_flush();
	mtr_shutdown();
#endif // PROFILING
}
void RawFileConverter()
{
	char* fileName = "fileToRead.raw";
	int width = 3840, height = 1920;
	const std::vector<uint16_t> &data = ReadingFiles(fileName, height, width);
	Loop(data, height, width);
}

void MatchKernel()
{
	int width = 1920, height = 1200;
	int channels;
	std::vector<uint8_t> data;
	if (!load(data, height, width, channels)) { throw "cannot load an image"; }
	MatchKernel_Result(data, height, width);
}
int main()
{
	RawFileConverter();
	//MatchKernel();
	return 0;
}

std::vector<uint16_t> ReadingFiles(char* fileName, int height, int width)
{
	FILE* rdFile = fopen(fileName, "rb+");
	std::vector<uint16_t> data;
	if (rdFile == 0) {
		printf("no file found!");
		return data; 
	}
	int size = height * width;
	data.resize(size);
	fread(reinterpret_cast<char*>(&data[0]), 2, size, rdFile);
	fclose(rdFile);
	return data;
}
