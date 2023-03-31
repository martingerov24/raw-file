#include "windowManager.h"
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

void WindowManager::bindTexture() {
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
}

void WindowManager::onNewFrame() {
    glfwPollEvents();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

bool WindowManager::shouldClose() {
    return !glfwWindowShouldClose(window);
}

void WindowManager::createContext() {
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

bool WindowManager::init() {
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    if (!glfwInit()) {
        throw "glfwInit() FAILED!";
    }

    window = glfwCreateWindow(800, 600, "Raw-File Viewer", NULL, NULL);

    if (!window) {
        glfwTerminate();
        return false;
    }

	createContext();
	glGenTextures(1, &texture);
	bindTexture();
    return window;
}

void WindowManager::terminate() {
    ImGui_ImplGlfw_Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    glfwDestroyWindow(window);
}

bool WindowManager::draw(const uint8_t* output, const ImageParams& params) {
    if(output==nullptr) {
        return false;
    }
    bool is_show = true;
    onNewFrame();
    ImGui::Begin("raw Image", &is_show);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, params.width, params.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, output);
    ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)), ImVec2(800, 600));
    ImGui::End();
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
    return true;
}