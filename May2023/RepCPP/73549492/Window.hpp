#pragma once

#include <string>
#include <memory>

#include "Singleton.hpp"





class GLFWwindow;


class Window : public Singleton<Window> {
friend class Input;

public:

static std::shared_ptr<Window> Create(unsigned int width, unsigned int height, std::string title,
bool fullscreen = false);

void UpdateViewport();
void SetViewport(int width, int height);
void MakeContextCurrent();
void Clear();
void SwapBuffers();
void PollEvents();
void Close();

void DrawElements(unsigned int count);

void GetSize(int &width, int &height) const;
bool IsClosed() const;

~Window();


private:

static void PreInit();
static void PostInit(GLFWwindow* window);

Window() {};


private:

GLFWwindow *mGlfwWindow = 0;

};


