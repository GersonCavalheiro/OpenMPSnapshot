

#ifndef CPP_RAYTRACING_RENDERER_HPP
#define CPP_RAYTRACING_RENDERER_HPP

#include <cmath>
#include <functional>
#include <memory>
#include <omp.h>

#include "../util.hpp"
#include "../values/color.hpp"
#include "../values/tensor.hpp"
#include "../world/entities/base.hpp"
#include "../world/geometries/base.hpp"
#include "../world/materials/base.hpp"
#include "../world/ray_segment.hpp"
#include "../world/scene.hpp"
#include "canvas.hpp"
#include "image.hpp"

namespace cpp_raytracing {


template <Dimension DIMENSION>
class Renderer {

public:

struct State {

RawImage& image;

unsigned long samples;
};

using RenderCallbackFunc = std::function<void(const State&)>;


constexpr static Scalar DEFAULT_RAY_MINIMAL_LENGTH = 1e-5;


constexpr static unsigned long DEFAULT_INFREQUENT_CALLBACK_FREQUENCY = 10;


Canvas canvas;


unsigned long samples = 1;

unsigned long ray_depth = 1;


Color ray_color_if_ray_ended = {0.0, 100.0, 0.0};

Color ray_color_if_no_background = {1.0, 1.0, 1.0};

Color ray_color_if_no_material = {1.0, 0.0, 1.0};


RenderCallbackFunc frequent_render_callback = nullptr;

RenderCallbackFunc infrequent_render_callback = nullptr;

unsigned long infrequent_callback_frequency =
DEFAULT_INFREQUENT_CALLBACK_FREQUENCY;


Scalar time = 0.0;


Scalar minimal_ray_length = DEFAULT_RAY_MINIMAL_LENGTH;


bool debug_normals = false;


Color ray_color_if_exterior_normal = {0.0, 0.0, 1.0};

Color ray_color_if_interior_normal = {1.0, 0.0, 0.0};


Renderer() = default;


Renderer(const Renderer&) = default;


Renderer(Renderer&&) = default;


Renderer& operator=(const Renderer&) = default;


Renderer& operator=(Renderer&&) = default;

virtual ~Renderer() = default;


virtual RawImage render(const Geometry<DIMENSION>& geometry,
Scene<DIMENSION>& scene) = 0;


Color ray_color(const Geometry<DIMENSION>& geometry,
const typename Scene<DIMENSION>::FreezeGuard& frozen_scene,
Ray<DIMENSION>* ray, const unsigned long depth) const {

using namespace tensor;

const std::optional<RaySegment<DIMENSION>> current_opt_segment =
ray->next_ray_segment();

if (!current_opt_segment.has_value()) {
return ray_color_if_ray_ended;
}
const RaySegment<DIMENSION>& current_segment =
current_opt_segment.value();

if (depth == 0) {
return background_color(geometry, frozen_scene, current_segment);
}

HitRecord<DIMENSION> record = frozen_scene.hit_record(
geometry, current_segment, minimal_ray_length);

if (!record.hits()) {
if (current_segment.is_infinite()) {
return background_color(geometry, frozen_scene,
current_segment);
}
return ray_color(geometry, frozen_scene, ray, depth - 1);
}

if (debug_normals) {
return record.front_face ? ray_color_if_exterior_normal
: ray_color_if_interior_normal;
}

if (!record.material) {
return ray_color_if_no_material;
}

const auto [onb_scatter_direction, color] =
record.material->scatter(record);
if (is_zero(onb_scatter_direction)) {
return color;
} else {
const auto from_onb_jacobian =
geometry.from_onb_jacobian(record.point);

const auto scattered_direction =
from_onb_jacobian * onb_scatter_direction;

std::unique_ptr<Ray<DIMENSION>> scattered_ray =
geometry.ray_from(record.point, scattered_direction);

return color * ray_color(geometry, frozen_scene,
scattered_ray.get(), depth - 1);
}
}

protected:

inline Color
background_color(const Geometry<DIMENSION>& geometry,
const typename Scene<DIMENSION>::FreezeGuard& frozen_scene,
const RaySegment<DIMENSION>& ray_segment) const {
if (frozen_scene.active_background == nullptr) {
return ray_color_if_no_background;
}
return frozen_scene.active_background->value(geometry, ray_segment);
}


inline void render_pixel_sample(
const unsigned long i, const unsigned long j,
const Geometry<DIMENSION>& geometry,
const typename Scene<DIMENSION>::FreezeGuard& frozen_scene,
RawImage& buffer) const {
Scalar x = Scalar(i) + random_scalar(-0.5, +0.5);
Scalar y = Scalar(j) + random_scalar(-0.5, +0.5);
x = (2.0 * x / static_cast<Scalar>(canvas.width) - 1.0);
y = (2.0 * y / static_cast<Scalar>(canvas.height) - 1.0);

std::unique_ptr<Ray<DIMENSION>> ray =
frozen_scene.active_camera.ray_for_coords(geometry, x, y);
const Color pixel_color =
ray_color(geometry, frozen_scene, ray.get(), ray_depth);
buffer[{i, j}] += pixel_color;
}
};


using Renderer3D = Renderer<Dimension{3}>;


template <Dimension DIMENSION>
class GlobalShutterRenderer : public Renderer<DIMENSION> {
public:

Scalar exposure_time = 0.0;


GlobalShutterRenderer() = default;


GlobalShutterRenderer(const GlobalShutterRenderer&) = default;


GlobalShutterRenderer(GlobalShutterRenderer&&) = default;


GlobalShutterRenderer& operator=(const GlobalShutterRenderer&) = default;


GlobalShutterRenderer& operator=(GlobalShutterRenderer&&) = default;

~GlobalShutterRenderer() override = default;

RawImage render(const Geometry<DIMENSION>& geometry,
Scene<DIMENSION>& scene) override {

RawImage buffer{this->canvas.width, this->canvas.height};

if (this->exposure_time == 0.0) {
const typename Scene<DIMENSION>::FreezeGuard& frozen_scene =
scene.freeze_for_time(random_scalar(
this->time, this->time + this->exposure_time));

for (unsigned long s = 1; s < this->samples + 1; ++s) {
render_sample(s, buffer, geometry, frozen_scene);
}

} else {
for (unsigned long s = 1; s < this->samples + 1; ++s) {

const typename Scene<DIMENSION>::FreezeGuard& frozen_scene =
scene.freeze_for_time(random_scalar(
this->time, this->time + this->exposure_time));

this->render_sample(s, buffer, geometry, frozen_scene);
}
}

buffer *= 1 / (Scalar(this->samples));
return buffer;
}

private:

inline void
render_sample(const unsigned long sample, RawImage& buffer,
const Geometry<DIMENSION>& geometry,
const typename Scene<DIMENSION>::FreezeGuard& frozen_scene) {

#pragma omp parallel for shared(buffer) schedule(static, 1)
for (unsigned long j = 0; j < this->canvas.height; ++j) {
for (unsigned long i = 0; i < this->canvas.width; ++i) {
this->render_pixel_sample(i, j, geometry, frozen_scene, buffer);
}
}

if (this->frequent_render_callback) {
this->frequent_render_callback(
typename Renderer<DIMENSION>::State{buffer, sample});
}
if (this->infrequent_render_callback &&
(sample % this->infrequent_callback_frequency == 0)) {
this->infrequent_render_callback(
typename Renderer<DIMENSION>::State{buffer, sample});
}
}
};


using GlobalShutterRenderer3D = GlobalShutterRenderer<Dimension{3}>;


template <Dimension DIMENSION>
class RollingShutterRenderer : public Renderer<DIMENSION> {
public:

Scalar frame_exposure_time = 0.0;


Scalar total_line_exposure_time = 0.0;


RollingShutterRenderer() = default;


RollingShutterRenderer(const RollingShutterRenderer&) = default;


RollingShutterRenderer(RollingShutterRenderer&&) = default;


RollingShutterRenderer& operator=(const RollingShutterRenderer&) = default;


RollingShutterRenderer& operator=(RollingShutterRenderer&&) = default;

~RollingShutterRenderer() override = default;
RawImage render(const Geometry<DIMENSION>& geometry,
Scene<DIMENSION>& scene) override {

RawImage buffer{this->canvas.width, this->canvas.height};

for (unsigned long s = 1; s < this->samples + 1; ++s) {
this->render_sample(s, buffer, geometry, scene);
}

buffer *= 1 / (Scalar(this->samples));
return buffer;
}

private:

inline void render_sample(const unsigned long sample, RawImage& buffer,
const Geometry<DIMENSION>& geometry,
Scene<DIMENSION>& scene) const {

for (unsigned long j = 0; j < this->canvas.height; ++j) {

const typename Scene<DIMENSION>::FreezeGuard& frozen_scene =
scene.freeze_for_time(mid_frame_time(j));

#pragma omp parallel for shared(buffer) schedule(static, 1)
for (unsigned long i = 0; i < this->canvas.width; ++i) {
this->render_pixel_sample(i, j, geometry, frozen_scene, buffer);
}
}

if (this->frequent_render_callback) {
this->frequent_render_callback(
typename Renderer<DIMENSION>::State{buffer, sample});
}
if (this->infrequent_render_callback &&
(sample % this->infrequent_callback_frequency == 0)) {
this->infrequent_render_callback(
typename Renderer<DIMENSION>::State{buffer, sample});
}
}


inline Scalar mid_frame_time(const unsigned horizonal_line) const {
auto res = this->time;
res += this->frame_exposure_time *
(Scalar(horizonal_line) / Scalar(this->canvas.height));
res += random_scalar(0.0, this->total_line_exposure_time);
return res;
}
};


using RollingShutterRenderer3D = RollingShutterRenderer<Dimension{3}>;

} 

#endif
