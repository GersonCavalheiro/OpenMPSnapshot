#pragma once

#include "glm/glm.hpp"
#include "Types.h"
#include "Scene.h"

#include "string"
#include "atlimage.h"

class CTracer
{
public:
SRay MakeRay(glm::uvec2 pixelPos);  
glm::vec3 TraceRay(SRay ray); 
glm::vec3 pickColor(float power);
void RenderImage(int xRes, int yRes, bool openMP);
void SaveImageToFile(std::string fileName);
CImage* LoadImageFromFile(std::string fileName);

SCamera m_camera;
CScene* m_scene;
};