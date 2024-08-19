

#pragma once

#include <GL/glew.h>
#include <string>

namespace tdogl {


class Shader { 
public:


static Shader shaderFromFile(const std::string& filePath, GLenum shaderType);



Shader(const std::string& shaderCode, GLenum shaderType);



GLuint object() const;

Shader(const Shader& other);
Shader& operator =(const Shader& other);
~Shader();

private:
GLuint _object;
unsigned* _refCount;

void _retain();
void _release();
};

}
