#pragma once
#include "./monolish_dense.hpp"
#include "./monolish_logger.hpp"
#include "./monolish_view1D.hpp"
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

#if USE_SXAT
#undef _HAS_CPP17
#endif
#include <random>
#if USE_SXAT
#define _HAS_CPP17 1
#endif

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {




template <typename Float> class vector {
private:

mutable bool gpu_status = false;

public:

std::shared_ptr<Float> val;


std::size_t val_nnz = 0;


std::size_t alloc_nnz = 0;


bool val_create_flag = false;

vector() { val_create_flag = true; }


vector(const size_t N);


vector(const size_t N, const Float value);


vector(const std::vector<Float> &vec);


vector(const std::initializer_list<Float> &list);


vector(const vector<Float> &vec);


vector(const view1D<vector<Float>, Float> &vec);


vector(const view1D<matrix::Dense<Float>, Float> &vec);


vector(const view1D<tensor::tensor_Dense<Float>, Float> &vec);


vector(const Float *start, const Float *end);


vector(const size_t N, const Float min, const Float max);


vector(const size_t N, const Float min, const Float max,
const std::uint32_t seed);


void send() const;


void recv();


void nonfree_recv();


void device_free() const;


[[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }


~vector() {
if (val_create_flag) {
if (get_device_mem_stat()) {
device_free();
}
}
}

[[nodiscard]] size_t get_offset() const { return 0; }



[[nodiscard]] const Float *data() const { return val.get(); }


[[nodiscard]] Float *data() { return val.get(); }


void resize(size_t N, Float Val = 0) {
if (get_device_mem_stat()) {
throw std::runtime_error("Error, GPU vector cant use resize");
}
if (val_create_flag) {
std::shared_ptr<Float> tmp(new Float[N], std::default_delete<Float[]>());
size_t copy_size = std::min(val_nnz, N);
for (size_t i = 0; i < copy_size; i++) {
tmp.get()[i] = data()[i];
}
for (size_t i = copy_size; i < N; i++) {
tmp.get()[i] = Val;
}
val = tmp;
alloc_nnz = N;
val_nnz = N;
} else {
throw std::runtime_error("Error, not create vector cant use resize");
}
}


void push_back(Float Val) {
if (get_device_mem_stat()) {
throw std::runtime_error("Error, GPU vector cant use push_back");
}
if (val_create_flag) {
if (val_nnz >= alloc_nnz) {
size_t tmp = val_nnz;
alloc_nnz = 2 * alloc_nnz + 1;
resize(alloc_nnz);
val_nnz = tmp;
}
data()[val_nnz] = Val;
val_nnz++;
} else {
throw std::runtime_error("Error, not create vector cant use push_back");
}
}

void move(const tensor::tensor_Dense<Float> &tensor_dense);

void move(const tensor::tensor_Dense<Float> &tensor_dense, int N);


[[nodiscard]] const Float *begin() const { return data(); }


[[nodiscard]] Float *begin() { return data(); }


[[nodiscard]] const Float *end() const { return data() + get_nnz(); }


[[nodiscard]] Float *end() { return data() + get_nnz(); }


[[nodiscard]] size_t size() const { return val_nnz; }


[[nodiscard]] size_t get_nnz() const { return val_nnz; }


void fill(Float value);


void print_all(bool force_cpu = false) const;


void print_all(std::string filename) const;



void operator=(const vector<Float> &vec);


void operator=(const view1D<vector<Float>, Float> &vec);


void operator=(const view1D<matrix::Dense<Float>, Float> &vec);


void operator=(const view1D<tensor::tensor_Dense<Float>, Float> &vec);


void operator=(const std::vector<Float> &vec);


[[nodiscard]] vector<Float> operator-();


[[nodiscard]] Float &operator[](size_t i) {
if (get_device_mem_stat()) {
throw std::runtime_error("Error, GPU vector cant use operator[]");
}
return data()[i];
}


bool equal(const vector<Float> &vec,
bool compare_cpu_and_device = false) const;


bool equal(const view1D<vector<Float>, Float> &vec,
bool compare_cpu_and_device = false) const;


bool equal(const view1D<matrix::Dense<Float>, Float> &vec,
bool compare_cpu_and_device = false) const;


bool equal(const view1D<tensor::tensor_Dense<Float>, Float> &vec,
bool compare_cpu_and_device = false) const;


bool operator==(const vector<Float> &vec) const;


bool operator==(const view1D<vector<Float>, Float> &vec) const;


bool operator==(const view1D<matrix::Dense<Float>, Float> &vec) const;


bool operator==(const view1D<tensor::tensor_Dense<Float>, Float> &vec) const;


bool operator!=(const vector<Float> &vec) const;


bool operator!=(const view1D<vector<Float>, Float> &vec) const;


bool operator!=(const view1D<matrix::Dense<Float>, Float> &vec) const;


bool operator!=(const view1D<tensor::tensor_Dense<Float>, Float> &vec) const;
};

} 
