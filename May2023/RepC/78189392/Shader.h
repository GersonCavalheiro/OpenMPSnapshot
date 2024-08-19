#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <GLEW\glew.h>
class Shader
{
public:
Shader();
void Init();
bool LoadShader(std::string filename, GLenum shader_type);
bool LinkProgram();
GLuint GetUniformLocation(const GLchar* name);
void UseProgram();
private:
std::string load_file(std::string filename);
void PrintShaderCompilationErrorInfo(int32_t shaderId);
void PrintShaderLinkingError(int32_t shaderId);
std::vector<GLuint> shaders;
GLuint shader_program;
};