#pragma once
#include "common.h"
#include <memory>
#include <vector>
enum ImageFormat { BMP, JPG, PNG };
enum WrappingOption { REPEAT, MIRRORED_REPEAT, CLAMP_TO_EDGE, CLAMP_TO_BORDER };
enum FilterOption { NEAREST, LINEAR };
class Image final {
size_t height, width;
std::vector<vec3> data;
WrappingOption wrapping;
FilterOption filter;
public:
Image(const size_t height, const size_t width,
WrappingOption wrapping = REPEAT, FilterOption filter = LINEAR)
: height(height), width(width),
data(height * width), wrapping{wrapping}, filter{filter} {}
Image(ImageFormat format, const std::string &location);
vec3 at(const float x, const float y) const;
const vec3 get(const size_t x, const size_t y) const noexcept {
return data[y * width + x];
}
void set(const size_t x, const size_t y, const vec3 &c) noexcept {
data[y * width + x] = c;
}
void write(ImageFormat format, const std::string &location) const;
};
