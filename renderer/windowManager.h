
#pragma once
#include <GLFW/glfw3.h>
#include "parameters.h"
class WindowManager {
    void bindTexture();
    void createContext();
    void onNewFrame();
public:
    WindowManager() = default;
    bool init();
    bool shouldClose();
    bool draw(const uint8_t* output, const ImageParams& params);
    void terminate();
private:
    GLFWwindow* window = nullptr;
    GLuint texture = -1;
};

