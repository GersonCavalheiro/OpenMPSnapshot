#pragma once
#include "objects/object.h"
class Plane : public Object {
vec3 point, normal;
vec3 colour;
public:
Plane(const vec3 &point, const vec3 &normal, const vec3 &colour)
: point{point}, normal{glm::normalize(normal)}, colour(colour) {}
~Plane() {}
bool intersects(const Ray &ray, float min_dist,
float max_dist) const override;
std::tuple<float, vec3, vec3> intersect(const Ray &ray,
float max_dist) const override;
};
