#pragma once

#include "Singleton.hpp"





class Shaders : public Singleton<Shaders> {

public:

typedef unsigned int Id;


public:

static std::shared_ptr<Shaders> CreateAndLoad();

void Attribute(std::string name, int stride, int offset);

~Shaders();


private:

void Load();


private:

Id mVertexShader = 0;
Id mFragmentShader = 0;
Id mProgram = 0;

};


