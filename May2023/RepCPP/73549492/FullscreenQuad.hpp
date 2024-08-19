#pragma once

#include "Singleton.hpp"





class FullscreenQuad : public Singleton<FullscreenQuad> {

public:

typedef unsigned int Id;


public:

static std::shared_ptr<FullscreenQuad> Create();

void Render();

~FullscreenQuad();


private:

void Load();


private:

Id mArrayObject = 0;
Id mVertexData = 0;
Id mIndices = 0;

};


