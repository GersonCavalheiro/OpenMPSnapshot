#include "Shaders.hpp"

#include <fstream>
#include <sstream>

#include <GL/glew.h>

#include "Error.hpp"
#include "Window.hpp"


std::shared_ptr<Shaders> Shaders::CreateAndLoad() {
auto shaders = std::make_shared<Shaders>();

shaders->Load();

SetSingleton(shaders);
return shaders;
}

void Shaders::Attribute(std::string name, int stride, int offset) {
GLuint pos = (GLuint) glGetAttribLocation(mProgram, name.c_str());
glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, stride, (const void *) offset);
glEnableVertexAttribArray(pos);
}

Shaders::~Shaders() {
if (mProgram) {
glDeleteProgram(mProgram);
}
if (mFragmentShader) {
glDeleteShader(mFragmentShader);
}
if (mVertexShader) {
glDeleteShader(mVertexShader);
}
}

void Shaders::Load() {
mVertexShader = glCreateShader(GL_VERTEX_SHADER);
mFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
mProgram = glCreateProgram();

#pragma omp parallel sections default(shared)
{
#pragma omp section
{
std::ifstream vertexFile("Vertex.glsl");
if (!vertexFile.is_open()) {
THROW_ERROR("Cannot open vertex shader file");
}

std::stringstream vertexStream;
vertexStream << vertexFile.rdbuf();
std::string vertexSource = vertexStream.str();
const char *vertexSourcePtr = vertexSource.c_str();

#pragma omp critical
{
Window::GetSingleton()->MakeContextCurrent();
glShaderSource(mVertexShader, 1, &vertexSourcePtr, nullptr);
}
}

#pragma omp section
{
std::ifstream fragmentFile("Fragment.glsl");
if (!fragmentFile.is_open()) {
THROW_ERROR("Cannot open fragment shader file");
}

std::stringstream fragmentStream;
fragmentStream << fragmentFile.rdbuf();
std::string fragmentSource = fragmentStream.str();
const char *fragmentSourcePtr = fragmentSource.c_str();

#pragma omp critical
{
Window::GetSingleton()->MakeContextCurrent();
glShaderSource(mFragmentShader, 1, &fragmentSourcePtr, nullptr);
}
}
}

glCompileShader(mVertexShader);
glCompileShader(mFragmentShader);
glAttachShader(mProgram, mVertexShader);
glAttachShader(mProgram, mFragmentShader);
glBindFragDataLocation(mProgram, 0, "oColor");
glLinkProgram(mProgram);
glUseProgram(mProgram);
}
