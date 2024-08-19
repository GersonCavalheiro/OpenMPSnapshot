#pragma once
#include "./monolish_logger.hpp"
#include <cassert>
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
template <typename Float> class vector;

namespace matrix {
template <typename Float> class Dense;
template <typename Float> class CRS;
template <typename Float> class LinearOperator;
} 




template <typename TYPE, typename Float> class view1D {
private:
TYPE &target;
Float *target_data;
size_t first;
size_t last;
size_t range;

public:

view1D(vector<Float> &x, const size_t start, const size_t size) : target(x) {
first = start;
last = start + size;
range = size;
target_data = x.data();
}


view1D(matrix::Dense<Float> &A, const size_t start, const size_t size)
: target(A) {
first = start;
last = start + size;
range = size;
target_data = A.data();
}


view1D(tensor::tensor_Dense<Float> &A, const size_t start, const size_t size)
: target(A) {
first = start;
last = start + size;
range = size;
target_data = A.data();
}


view1D(view1D<vector<Float>, Float> &x, const size_t start, const size_t size)
: target(x) {
first = x.get_first() + start;
last = first + size;
range = size;
target_data = x.data();
}


view1D(view1D<matrix::Dense<Float>, Float> &x, const size_t start,
const size_t size)
: target(x) {
first = x.get_first() + start;
last = first + size;
range = size;
target_data = x.data();
}


view1D(view1D<tensor::tensor_Dense<Float>, Float> &x, const size_t start,
const size_t size)
: target(x) {
first = x.get_first() + start;
last = first + size;
range = size;
target_data = x.data();
}


[[nodiscard]] size_t size() const { return range; }


[[nodiscard]] size_t get_nnz() const { return range; }


[[nodiscard]] size_t get_first() const { return first; }


[[nodiscard]] size_t get_last() const { return last; }


[[nodiscard]] size_t get_offset() const { return first; }


void set_first(size_t i) { first = i; }


void set_last(size_t i) {
assert(first + i <= target.get_nnz());
last = i;
}


[[nodiscard]] size_t get_device_mem_stat() const {
return target.get_device_mem_stat();
}


[[nodiscard]] Float *data() const { return target_data; }


[[nodiscard]] Float *data() { return target_data; }


[[nodiscard]] Float *begin() const { return target_data + get_offset(); }


[[nodiscard]] Float *begin() { return target_data + get_offset(); }


[[nodiscard]] Float *end() const { return target_data + range; }


[[nodiscard]] Float *end() { return target_data + range; }


void print_all(bool force_cpu = false) const;


void resize(size_t N) {
assert(first + N <= target.get_nnz());
last = first + N;
}


[[nodiscard]] Float &operator[](const size_t i) {
if (target.get_device_mem_stat()) {
throw std::runtime_error("Error, GPU vector cant use operator[]");
}
return target_data[i + first];
}


void fill(Float value);
};


} 
