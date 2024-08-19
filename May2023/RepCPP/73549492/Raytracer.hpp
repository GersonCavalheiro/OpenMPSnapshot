#pragma once

#include <glm/glm.hpp>

#include "Singleton.hpp"





class Raytracer : public Singleton<Raytracer> {

public:

enum class Object {
None,
Ground,
Battery,
BatteryCore
};


public:

static std::shared_ptr<Raytracer> Create();

glm::vec3 TracePixel(const glm::vec2 &uv, float time) const;


private:

void MakeRay(float time, const glm::vec2 &uvNorm, const glm::vec3 &cameraTarget, glm::vec3 &rayOrigin,
glm::vec3 &rayDirection) const;
glm::vec3 ShootRay(float time, const glm::vec3 &rayOrigin, const glm::vec3 &rayDirection) const;
float Map(float time, const glm::vec3 &point, Raytracer::Object &object) const;
glm::vec3 MapNormal(float time, const glm::vec3 &point) const;
float Shadow(float time, const glm::vec3 &point, const glm::vec3 normal)const;


private:

const glm::vec3 mBackgroundColor = glm::vec3(.01f, .01f, .015f);
const glm::vec3 mAmbient = glm::vec3(.01f);
const glm::vec3 mSun = glm::normalize(glm::vec3(-.8f, 1, .4f));

};


