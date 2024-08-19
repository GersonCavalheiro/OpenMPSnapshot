#pragma once
#include "Math.hpp"
#include "Ray.hpp"
#include "Color.hpp"
#include "Camera.hpp"
#include "Lights.hpp"
#include "Objects.hpp"
#include "Scene.hpp"
#include "FrameBuffer.hpp"


class RayTracer {
public:
RayTracer();

void traceScene(const Camera& camera, const Scene& scene, FrameBuffer& frameBuffer) const;

float  bias()              const;
size_t maxNumReflections() const;
Color  shadowColor()       const;
Color  backgroundColor()   const;

void setBias(float bias);
void setMaxNumReflections(size_t maxNumReflections);
void setShadowColor(const Color& shadowColor);
void setBackgroundColor(const Color& backgroundColor);

Ray reflectRay(const Ray& ray, const Intersection& intersection) const;
bool findNearestIntersection(const Camera& camera, const Scene& scene, const Ray& ray, Intersection& result) const;

private:
float bias_;
size_t maxNumReflections_;
Color shadowColor_;
Color backgroundColor_;

static constexpr float  DEFAULT_BIAS = 1e-02f;
static constexpr size_t DEFAULT_MAX_NUM_REFLECTIONS = 3;
static constexpr Color  DEFAULT_SHADOW_COLOR     { 0.125f, 0.125f, 0.125f };
static constexpr Color  DEFAULT_BACKGROUND_COLOR { 0.500f, 0.500f, 0.500f };

Color traceRay(const Camera& camera, const Scene& scene, const Ray& ray, size_t depth) const;

bool isInShadow(const Camera& camera, const Intersection& intersection, const ILight& light, const Scene& scene) const;

Color computeDiffuseColor(const Intersection& intersection, const ILight& light) const;
Color computeSpecularColor(const Intersection& intersection, const ILight& light, const Camera& camera) const;
};
std::ostream& operator<<(std::ostream& os, const RayTracer& tracer);
