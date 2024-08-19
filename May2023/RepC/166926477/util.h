#pragma once
#include "objects/model.h"
#include "ray.h"
#include <vector>
RawOBJ loadOBJ(const std::string &obj_file_name);
bool intersectsAABB(const Ray &ray, const vec3 &p1, const vec3 &p2,
const float min_dist = 0.0f,
const float max_dist = inf) noexcept;
