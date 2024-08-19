#pragma once

#include "Types.h"

class CTracer;
class CScene
{
public:
CScene();
~CScene();
bool load(FILE *path);
void preprocess();
void processSignalMap(glm::vec3 routerPos, float routerPower, int maxRays);
void boxFilter();
bool countIntersection(SRay ray, glm::uvec3 triangle, float *distance);
const unsigned int voxelSize;
friend class CTracer;
private:
void marchRay(SRay ray, float routerPower);

SMesh flat;
glm::vec3 lowBound, highBound;
glm::vec3 resizeRatio;
float ***voxel;
};