#pragma once
#include "ray.h"
#include <utility>
#include <vector>
class Scene;
class Object {
protected:
Scene *scene = nullptr;
public:
Object() {}
virtual ~Object() noexcept {}
virtual bool intersects(const Ray &ray, float min_dist,
float max_dist) const = 0;
virtual std::tuple<float, vec3, vec3> intersect(const Ray &ray,
float max_dist) const = 0;
Scene *getScene() const noexcept { return scene; }
void setScene(Scene *scene) noexcept { this->scene = scene; }
};
