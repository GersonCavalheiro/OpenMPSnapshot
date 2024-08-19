#pragma once

#include <stdint.h>
#include <string>
#include <fstream>
#include <CL/sycl.hpp>

#include "TriMesh.h"

#include <glm/glm.hpp>

template<typename trimeshtype>
inline glm::vec3 trimesh_to_glm(trimeshtype a) {
return glm::vec3(a[0], a[1], a[2]);
}

template<typename trimeshtype>
inline trimeshtype glm_to_trimesh(glm::vec3 a) {
return trimeshtype(a[0], a[1], a[2]);
}

inline bool checkVoxel(size_t x, size_t y, size_t z, const glm::uvec3 gridsize, const unsigned int* vtable) {
size_t location = x + (y*gridsize.y) + (z*gridsize.y*gridsize.z);
size_t int_location = location / size_t(32);
unsigned int bit_pos = size_t(31) - (location % size_t(32)); 
if ((vtable[int_location]) & (1 << bit_pos)){
return true;
}
return false;
}

template <typename T>
struct AABox {
T min;
T max;
AABox() : min(T()), max(T()) {}
AABox(T min, T max) : min(min), max(max) {}
};

struct voxinfo {
AABox<glm::vec3> bbox;
glm::uvec3 gridsize;
size_t n_triangles;
glm::vec3 unit;

voxinfo(AABox<glm::vec3> bbox, glm::uvec3 gridsize, size_t n_triangles)
: gridsize(gridsize), bbox(bbox), n_triangles(n_triangles) {
unit.x = (bbox.max.x - bbox.min.x) / float(gridsize.x);
unit.y = (bbox.max.y - bbox.min.y) / float(gridsize.y);
unit.z = (bbox.max.z - bbox.min.z) / float(gridsize.z);
}

void print() {
printf("[Voxelization] Bounding Box: (%f,%f,%f)-(%f,%f,%f) \n", 
bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z);
printf("[Voxelization] Grid size: %i %i %i \n", gridsize.x, gridsize.y, gridsize.z);
printf("[Voxelization] Triangles: %zu \n", n_triangles);
printf("[Voxelization] Unit length: x: %f y: %f z: %f\n", unit.x, unit.y, unit.z);
}
};

template <typename T>
inline AABox<T> createMeshBBCube(AABox<T> box) {
AABox<T> answer(box.min, box.max); 
glm::vec3 lengths = box.max - box.min; 
float max_length = glm::max(lengths.x, glm::max(lengths.y, lengths.z)); 
for (unsigned int i = 0; i < 3; i++) { 
if (max_length == lengths[i]){
continue;
} else {
float delta = max_length - lengths[i]; 
answer.min[i] = box.min[i] - (delta / 2.0f); 
answer.max[i] = box.max[i] + (delta / 2.0f); 
}
}

glm::vec3 epsilon = (answer.max - answer.min) / 10001.0f;
answer.min -= epsilon;
answer.max += epsilon;
return answer;
}

void inline printBits(size_t const size, void const * const ptr) {
unsigned char *b = (unsigned char*)ptr;
unsigned char byte;
int i, j;
for (i = static_cast<int>(size) - 1; i >= 0; i--) {
for (j = 7; j >= 0; j--) {
byte = b[i] & (1 << j);
byte >>= j;
if (byte) {
printf("X");
}
else {
printf(".");
}
}
}
puts("");
}

inline std::string readableSize(size_t bytes) {
double bytes_d = static_cast<double>(bytes);
std::string r;
if (bytes_d <= 0) r = "0 Bytes";
else if (bytes_d >= 1099511627776.0) r = std::to_string(static_cast<size_t>(bytes_d / 1099511627776.0)) + " TB";
else if (bytes_d >= 1073741824.0) r = std::to_string(static_cast<size_t>(bytes_d / 1073741824.0)) + " GB";
else if (bytes_d >= 1048576.0) r = std::to_string(static_cast<size_t>(bytes_d / 1048576.0)) + " MB";
else if (bytes_d >= 1024.0) r = std::to_string(static_cast<size_t>(bytes_d / 1024.0)) + " KB";
else r = std::to_string(static_cast<size_t>(bytes_d)) + " bytes";
return r;
};

inline bool file_exists(const std::string& name) {
std::ifstream f(name.c_str());
return f.good();
}
