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

//extern std::vector<uint8_t> result;
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
	std::vector<uint8_t> h_result;
	int size = height * width * 3;
	h_result.resize(size);

	Cuda cuda(h_result, data, height, width);
	cuda.startup(size);

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

	cuda.sync();
	createContext(window);

	bool is_show = true;
	GLuint texture;
	glGenTextures(1, &texture);

	while (!glfwWindowShouldClose(window))
	{
		cuda.rawValue();
		onNewFrame();
		ImGui::Begin("raw Image", &is_show);
		bindTexture(texture);
		cuda.sync();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, h_result.data());
		ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)), ImVec2(800, 600));
		ImGui::End();
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glDeleteTextures(sizeof(texture), &texture);
		glfwSwapBuffers(window);
	}

	cuda.outPutFile();
	ImGui_ImplGlfw_Shutdown();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
	glfwDestroyWindow(window);
}

void MatchKernel_Result(const std::vector<uint8_t>& data, const int height, const int width)
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	std::vector<uint16_t> h_result;

	int size = height * width;
	std::vector<descriptor_t> query;
	std::vector<descriptor_t> train;

	CudaKeypoints cuda(data, height, width);
	cuda.startup(size, leftPoint.size() / 2);

	cuda.cudaUploadKeypoints(leftPoint);
	cuda.Kernel(leftPoint.size() / 2);
	cuda.cudaMemcpyD2H(query, leftPoint.size() / 2);
	cuda.sync(stream);

	cuda.cudaUploadKeypoints(rightPoint);
	cuda.Kernel(rightPoint.size() / 2);
	cuda.cudaMemcpyD2H(train, rightPoint.size() / 2);
	cuda.sync(stream);

	auto srt = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	auto time = end - srt;
	cuda.MemoryAllocationAsync(stream, query, train);
	for (int i = 0; i < 1000; i++)
	{
		srt = std::chrono::high_resolution_clock::now();
		cuda.MemcpyUploadAsyncForMatches(stream, query, train);
		cuda.match_gpu_caller(stream, query.size(), train.size());
		cuda.downloadAsync(stream, h_result, query.size());
		end = std::chrono::high_resolution_clock::now();
		time += end - srt;
	}
	cuda.sync(stream);
	cuda.cudaFreeAcyncMatcher(stream);

	printf("%d\n", std::chrono::duration_cast<std::chrono::milliseconds>(time).count());
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
	//RawFileConverter();
	//KeypointTest();
	MatchKernel();
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
