#pragma once

#include "opencl.hpp"

#include <opencv2/opencv.hpp>

#include <array>

template<typename Element>
class CLMat {

cv::Mat_<Element>* mat;
cl::Image2D buffer;

public:
CLMat()
: mat(nullptr) {}

CLMat(size_t rows,
size_t cols,
cl::Image2D buffer,
cl_bool blocking             = CL_TRUE,
cl_map_flags flags           = CL_MAP_READ,
std::array<size_t, 3> origin = {{0, 0, 0}},
std::array<size_t, 3> region = {{0, 0, 0}})
: buffer(buffer) {

if(region == std::array<size_t, 3>{{0, 0, 0}}) {
region = {{cols, rows, 1}};
}

Element* mapped_memory = reinterpret_cast<Element*>(
cl_singletons::queue.enqueueMapImage(buffer,
blocking,
flags,
reinterpret_cast<cl::size_t<3>&>(origin),
reinterpret_cast<cl::size_t<3>&>(region),
NULL,
NULL));

mat = new cv::Mat_<Element>(rows, cols, mapped_memory);
}

CLMat(CLMat const& other) = delete;
CLMat(CLMat&& other) {
destroyMyself();
buffer    = std::move(other.buffer);
mat       = other.mat;
other.mat = nullptr;
}

CLMat& operator=(CLMat const& other) = delete;
CLMat& operator                      =(CLMat&& other) {
if(&other != this) {
destroyMyself();
buffer    = std::move(buffer);
mat       = other.mat;
other.mat = nullptr;
}

return *this;
}

~CLMat() {
destroyMyself();
}

cv::Mat_<Element>& get() {
return *mat;
}

cv::Mat_<Element> const& get() const {
return *mat;
}

private:
void destroyMyself() {
if(mat != nullptr) {
cl_singletons::queue.enqueueUnmapMemObject(buffer, mat->data);
delete mat;
}
}
};
