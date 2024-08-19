#pragma once
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include "monolish_vector.hpp"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace tensor {
template <typename Float> class tensor_Dense;
template <typename Float> class tensor_COO {
private:

std::vector<size_t> shape;


mutable bool gpu_status = false;

public:

std::vector<std::vector<size_t>> index;


std::shared_ptr<Float> val;


size_t val_nnz = 0;


std::size_t alloc_nnz = 0;


bool val_create_flag = false;

tensor_COO() : shape(), gpu_status(false), index(), val_nnz(0) {
val_create_flag = true;
}


tensor_COO(const std::vector<size_t> &shape_)
: shape(shape_), gpu_status(false), index(), val_nnz(0) {
val_create_flag = true;
}


void convert(const tensor::tensor_Dense<Float> &tens);


tensor_COO(const tensor::tensor_Dense<Float> &tens) {
val_create_flag = true;
convert(tens);
}


tensor_COO(const std::vector<size_t> &shape_,
const std::vector<std::vector<size_t>> &index_,
const Float *value);


tensor_COO(const tensor_COO<Float> &coo);


tensor_COO(const tensor_COO<Float> &coo, Float value);


void print_all(bool force_cpu = false) const;


void print_all(const std::string filename) const;


[[nodiscard]] double get_data_size() const {
return get_nnz() * sizeof(Float) / 1.0e+9;
}


[[nodiscard]] Float at(const std::vector<size_t> &pos) const;


[[nodiscard]] Float at(const std::vector<size_t> &pos) {
return static_cast<const tensor_COO *>(this)->at(pos);
};


void set_ptr(const std::vector<size_t> &shape,
const std::vector<std::vector<size_t>> &index,
const std::vector<Float> &v);


void set_ptr(const std::vector<size_t> &shape,
const std::vector<std::vector<size_t>> &index,
const size_t vsize, const Float *v);


[[nodiscard]] std::vector<size_t> get_shape() const { return shape; }


[[nodiscard]] size_t get_nnz() const { return val_nnz; }


void fill(Float value);


void set_shape(const std::vector<size_t> &shape) { this->shape = shape; }


[[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }


[[nodiscard]] const Float *data() const { return val.get(); }


[[nodiscard]] Float *data() { return val.get(); }


void resize(const size_t N, Float Val = 0) {
if (get_device_mem_stat()) {
throw std::runtime_error("Error, GPU matrix cant use resize");
}
if (val_create_flag) {
std::shared_ptr<Float> tmp(new Float[N], std::default_delete<Float[]>());
size_t copy_size = std::min(val_nnz, N);
for (size_t i = 0; i < copy_size; ++i) {
tmp.get()[i] = data()[i];
}
for (size_t i = copy_size; i < N; ++i) {
tmp.get()[i] = Val;
}
val = tmp;
alloc_nnz = N;
val_nnz = N;

index.resize(N);
} else {
throw std::runtime_error("Error, not create vector cant use resize");
}
}


[[nodiscard]] std::string type() const { return "tensor_COO"; }


[[nodiscard]] const Float *begin() const { return data(); }


[[nodiscard]] Float *begin() { return data(); }


[[nodiscard]] const Float *end() const { return data() + get_nnz(); }


[[nodiscard]] Float *end() { return data() + get_nnz(); }


void diag(vector<Float> &vec) const;
void diag(view1D<vector<Float>, Float> &vec) const;
void diag(view1D<matrix::Dense<Float>, Float> &vec) const;
void diag(view1D<tensor::tensor_Dense<Float>, Float> &vec) const;


void operator=(const tensor_COO<Float> &tens);


[[nodiscard]] Float &operator[](size_t i) {
if (get_device_mem_stat()) {
throw std::runtime_error("Error, GPU vector cant use operator[]");
}
return data()[i];
}


[[nodiscard]] bool equal(const tensor_COO<Float> &tens,
bool compare_cpu_and_device = false) const;


[[nodiscard]] bool operator==(const tensor_COO<Float> &tens) const;


[[nodiscard]] bool operator!=(const tensor_COO<Float> &tens) const;


size_t get_index(const std::vector<size_t> &pos) {
if (pos.size() != this->shape.size()) {
throw std::runtime_error("pos size should be same with the shape");
}
size_t ind = 0;
for (auto i = 0; i < pos.size(); ++i) {
ind *= this->shape[i];
ind += pos[i];
}
return ind;
}


std::vector<size_t> get_index(const size_t pos) {
std::vector<size_t> ind(this->shape.size(), 0);
auto pos_copy = pos;
for (int i = (int)this->shape.size() - 1; i >= 0; --i) {
ind[i] = pos_copy % this->shape[i];
pos_copy /= this->shape[i];
}
return ind;
}


void insert(const std::vector<size_t> &pos, const Float val);

private:
void _q_sort(int lo, int hi);

public:

void sort(bool merge);
};
} 
} 
