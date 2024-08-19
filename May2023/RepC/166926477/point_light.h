#pragma once
#include "common.h"
#include "light.h"
class PointLight : public Light {
vec3 colour;
public:
PointLight(const vec3 &loc, const vec3 &colour)
: Light{loc}, colour(colour) {}
vec3 get_colour(const vec3 &from) const override { return colour; }
};
