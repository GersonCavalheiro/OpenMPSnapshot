#pragma once
#include "common.h"
class Scene;
class Light {
Scene *scene = nullptr;
vec3 pos;
public:
explicit Light(const vec3 &pos) : pos(pos) {}
virtual ~Light() {}
vec3 get_position() const noexcept { return pos; }
virtual vec3 get_colour(const vec3 &from) const = 0;
Scene *get_scene() const noexcept { return scene; }
void set_scene(Scene *scene) noexcept { this->scene = scene; }
};
