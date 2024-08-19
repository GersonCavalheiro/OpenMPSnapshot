#pragma once
#include "lights/light.h"
#include "objects/object.h"
#include "ray.h"
#include <memory>
#include <vector>
class Scene {
std::vector<std::unique_ptr<Object>> objects;
std::vector<std::unique_ptr<Light>> lights;
float max_dist = inf;
vec3 background;
public:
Scene() {}
explicit Scene(std::vector<std::unique_ptr<Object>> &&objects,
std::vector<std::unique_ptr<Light>> &&lights)
: objects{std::move(objects)}, lights{std::move(lights)} {}
~Scene() {}
void addObject(std::unique_ptr<Object> &&object) {
objects.push_back(std::move(object));
}
void addLight(std::unique_ptr<Light> &&light) {
lights.push_back(std::move(light));
}
vec3 intersect(const Ray &ray, size_t depth) const;
};
