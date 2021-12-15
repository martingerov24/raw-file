#include <stdio.h>
#include <vector>
#include <string>

#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "build/CudaClass.h"


//------------------------------------------------Predifining Functions
std::vector<uint16_t> ReadingFiles(char* fileName, int& height, int& width);
//------------------------------------------------Color operations
//__device__
//void Color(uint16_t number, uint8_t& n)
//{
//	// this is the solution if little indian
//	uint8_t first_8_bits = number & 0b11111111; first_8_bits = first_8_bits >> 4;
//	number = number >> 8;
//	n = number & 0b11111111; n = n >> 4;
//	// so now we have smth like bit1 -> 10101011, bit0 -> 10101011;
//	n = n & 0b1111; // basicly the paddings are throun away
//	// now we have 2 4 bit numbers and when combining them OR || XOR
//	n = n << 4;
//	n |= first_8_bits;
//	// the second number is our putput
//}
//
//__global__ void Checker(uint16_t* d_Data, uint8_t* cpy_Data, int width, int height)
//{
//	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//	int y = (blockIdx.y * blockDim.y) + threadIdx.y; 
//	if (x < width && x >= 0 
//		&& y < height && y>=0)
//	{
//		int calc = y * width + x;  //their scope is threadLifeTime
//		uint8_t n = 0;
//		Color(d_Data[calc], n);
//		//h   !w
//		short idx = (y & 1) + !(x & 1);
//		cpy_Data[3 * calc + 0] = 0;//r
//		cpy_Data[3 * calc + 1] = 0;//g
//		cpy_Data[3 * calc + 2] = 0;//b
//		cpy_Data[3 * calc + idx] = n;
//	}
//}
////TODO::clean up the code, imo create a class for opengl and split the GetCudaRdy to more functions
//__host__
//void GetCudaRdy(std::vector<uint8_t> &h_cpy, const std::vector<uint16_t>& data, const int& height, const int& width)
//{
//	int size = height * width;
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//	uint16_t* d_data;
//	uint8_t* cpyData;
//
//	cudaStream_t stream;
//	cudaError_t cudaStatus = cudaError_t(0);
//	cudaStatus = cudaSetDevice(0);
//	assert(cudaStatus == cudaSuccess, "you do not have cuda capable device!");
//	cudaStatus = cudaStreamCreate(&stream);
//
//	cudaStatus = cudaMalloc((void**)&d_data, data.size() * sizeof(uint16_t));//sizeof(uint16_t) * size
//	assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");
//
//	cudaStatus = cudaMalloc((void**)&cpyData, sizeof(uint8_t) * h_cpy.size() * 3);
//	assert(cudaStatus == cudaSuccess, "cudaMalloc failed!");
//	// learn to get stram and make the function async
//	auto start = std::chrono::high_resolution_clock::now();
//	
//	cudaStatus = cudaMemcpyAsync(d_data, data.data(), sizeof(uint16_t) * size, cudaMemcpyHostToDevice, stream);
//	assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");
//
//	cudaStatus = cudaMemcpyAsync(cpyData, h_cpy.data(), sizeof(uint8_t) * size * 3, cudaMemcpyHostToDevice, stream);
//	assert(cudaStatus == cudaSuccess, "not able to tansfer Data!");// here i am actually not in need to transfer data, but i wanted to see if it makes difference
//	dim3 sizeOfBlock(((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), height); // 4 , 2
//
//	Checker << <sizeOfBlock, THREADS_PER_BLOCK, 0, stream >> > (d_data, cpyData, width, height);
//
//	cudaStatus = cudaMemcpyAsync(h_cpy.data(), cpyData, sizeof(uint8_t) * size * 3, cudaMemcpyDeviceToHost, stream);
//	cudaStatus = cudaStreamSynchronize(stream);
//	//cudaStatus = cudaDeviceSynchronize();
//	
//	auto end = std::chrono::high_resolution_clock::now();
//	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//	printf("%d -> is the time measured", duration);
//
//	cudaFree(cpyData);
//	cudaFree(d_data);// it was said to -> cudaFree ( void* devPtr )Frees memory on the device.
//	cudaStreamDestroy(stream);
//}

void Loop(std::vector<uint8_t>& h_cpy, const std::vector<uint16_t>& data, const int& height, const int& width)
{
	int size = height * width;
	
	uint16_t* d_data;
	uint8_t* cpyData;
	Cuda cuda(h_cpy, data, height, width, d_data, cpyData);
	cuda.startup(size);

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	if (!glfwInit()) {
		throw "glfwInit() FAILED!";
	}

	GLFWwindow* window = glfwCreateWindow(640, 480, "My Title", NULL, NULL);

	if (!window) {
		glfwTerminate();
		throw "no window created";
	}

	cuda.sync();
	
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	bool is_show = true;
	bool oneImage = false;
	bool twoImages = false;
	GLuint texture;
	cuda.rawValue();
	cuda.sync();

	while (!glfwWindowShouldClose(window))
	{
		cuda.rawValue();
		glfwPollEvents();
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("imgui image", &is_show);
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
		cuda.sync();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, height, width, 0, GL_RGB, GL_UNSIGNED_BYTE, h_cpy.data());
		ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)), ImVec2(height, width));
		ImGui::End();

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
	cuda.~Cuda();
}
bool IfFileCanOpen(const char* fileName, const std::string width, const std::string height) {
	FILE* file = fopen(fileName, "r");
	if (file == 0)
	{
		printf("not able to open the file for reading!\n");
		return false;
	}
	fclose(file);
	for (int i = 0; i < fmax(width.size(), height.size()); i++)
	{
		if (width[i] > '9' || width[i] < '0'
			|| height[i] > '9' || height[i] < '0')
		{
			printf("you have declared invalid numbers for width and height\n");
			return false;
		}
	}
	return true;
}
void guiClearColor()
{
	glfwPollEvents();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
}
int main()
{
	char* fileName = "fileToRead.raw";
	int width = 3840, height = 1920;
	const std::vector<uint16_t> &data = ReadingFiles(fileName, height, width);// reading the 

	std::vector<uint8_t> h_cpy;
	h_cpy.resize(width * height * 3);

	Loop(h_cpy, data, height, width);
	return 0;
}

std::vector<uint16_t> ReadingFiles(char* fileName, int& height, int& width)
{
	FILE* rdFile = fopen(fileName, "rb+");
	std::vector<uint16_t> data;
	if (rdFile == 0) {
		printf("no file found!");
		return data; 
	}
	data.resize(height * width);
	fread(reinterpret_cast<char*>(&data[0]), 2, height * width, rdFile);
	fclose(rdFile);
	return data;
}
