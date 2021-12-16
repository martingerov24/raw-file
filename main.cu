#include <stdio.h>
#include <vector>
#include <string>

#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "build/CudaClass.h"

std::vector<uint16_t> ReadingFiles(char* fileName, int& height, int& width);

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
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, h_cpy.data());
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
	cuda.~Cuda();
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
