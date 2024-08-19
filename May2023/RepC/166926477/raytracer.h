#pragma once
#include "camera.h"
#include "image.h"
#include "scene.h"
#include <memory>
#include <string>
class Scene;
class RayTracer {
size_t screen_width = 320, screen_height = 200;
float horizontal_fov = 60;
size_t max_depth = 100;
size_t aa_num = 1;
public:
RayTracer() noexcept {} 
RayTracer(size_t screen_width, size_t screen_height,
float horizontal_fov) noexcept
: screen_width{screen_width}, screen_height{screen_height},
horizontal_fov{horizontal_fov} {}
~RayTracer() noexcept {}
Image trace(const Camera &camera, const Scene &scene,
const std::string &render_name);
};
