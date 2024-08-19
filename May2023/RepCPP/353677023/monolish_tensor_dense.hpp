#pragma once
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include "monolish_vector.hpp"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace tensor {
template <typename Float> class tensor_COO;
template <typename Float> class tensor_Dense {
private:

std::vector<size_t> shape;


mutable bool gpu_status = false;

public:

std::shared_ptr<Float> val;


size_t val_nnz = 0;


size_t alloc_nnz = 0;


bool val_create_flag = false;

tensor_Dense() { val_create_flag = true; }


void convert(const tensor::tensor_Dense<Float> &tens);


tensor_Dense(const tensor::tensor_Dense<Float> &tens);


void convert(const tensor::tensor_COO<Float> &tens);


tensor_Dense(const tensor::tensor_COO<Float> &tens) {
convert(tens);
val_create_flag = true;
}


void convert(const matrix::Dense<Float> &dense);


tensor_Dense(const matrix::Dense<Float> &dense) {
convert(dense);
val_create_flag = true;
}


void convert(const vector<Float> &vec);


tensor_Dense(const vector<Float> &vec) {
convert(vec);
val_create_flag = true;
}


tensor_Dense(const std::vector<size_t> &shape);


tensor_Dense(const std::vector<size_t> &shape, const Float *value);


tensor_Dense(const std::vector<size_t> &shape,
const std::vector<Float> &value);


tensor_Dense(const std::vector<size_t> &shape, const Float min,
const Float max);


tensor_Dense(const std::vector<size_t> &shape, const Float min,
const Float max, const std::uint32_t seed);


tensor_Dense(const tensor_Dense<Float> &tens, Float value);


void set_ptr(const std::vector<size_t> &shape,
const std::vector<Float> &value);


void set_ptr(const std::vector<size_t> &shape, const Float *value);


[[nodiscard]] std::vector<size_t> get_shape() const { return shape; }


[[nodiscard]] size_t get_nnz() const { return val_nnz; }


void set_shape(const std::vector<size_t> &shape) { this->shape = shape; };


[[nodiscard]] std::string type() const { return "tensor_Dense"; }


[[nodiscard]] double get_data_size() const {
return get_nnz() * sizeof(Float) / 1.0e+9;
}


[[nodiscard]] Float at(const std::vector<size_t> &pos) const;


template <typename... Args>
[[nodiscard]] Float at(const std::vector<size_t> &pos, const size_t dim,
const Args... args) const {
std::vector<size_t> pos_copy = pos;
pos_copy.push_back(dim);
return this->at(pos_copy, args...);
};


template <typename... Args>
[[nodiscard]] Float at(const size_t dim, const Args... args) const {
std::vector<size_t> pos(1);
pos[0] = dim;
return this->at(pos, args...);
};


[[nodiscard]] Float at(const std::vector<size_t> &pos) {
return static_cast<const tensor_Dense *>(this)->at(pos);
};


template <typename... Args>
[[nodiscard]] Float at(const std::vector<size_t> &pos, const Args... args) {
return static_cast<const tensor_Dense *>(this)->at(pos, args...);
};


template <typename... Args>
[[nodiscard]] Float at(const size_t dim, const Args... args) {
return static_cast<const tensor_Dense *>(this)->at(dim, args...);
};


void insert(const std::vector<size_t> &pos, const Float Val);


void print_all(bool force_cpu = false) const;


void send() const;


void recv();


void nonfree_recv();


void device_free() const;


[[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }


~tensor_Dense() {
if (val_create_flag) {
if (get_device_mem_stat()) {
device_free();
}
}
}


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
} else {
throw std::runtime_error("Error, not create vector cant use resize");
}
}


void resize(const std::vector<size_t> &shape, Float val = 0) {
size_t N = 1;
for (auto n : shape) {
N *= n;
}
resize(N, val);
this->shape = shape;
}


void move(const matrix::Dense<Float> &dense);


void move(const vector<Float> &vec);


[[nodiscard]] const Float *begin() const { return data(); }


[[nodiscard]] Float *begin() { return data(); }


[[nodiscard]] const Float *end() const { return data() + get_nnz(); }


[[nodiscard]] Float *end() { return data() + get_nnz(); }


void fill(Float value);


void operator=(const tensor_Dense<Float> &tens);


[[nodiscard]] Float &operator[](size_t i) {
if (get_device_mem_stat()) {
throw std::runtime_error("Error, GPU vector cant use operator[]");
}
return data()[i];
}


[[nodiscard]] bool equal(const tensor_Dense<Float> &tens,
bool compare_cpu_and_device = false) const;


[[nodiscard]] bool operator==(const tensor_Dense<Float> &tens) const;


[[nodiscard]] bool operator!=(const tensor_Dense<Float> &tens) const;


size_t get_index(const std::vector<size_t> &pos) const {
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


std::vector<size_t> get_index(const size_t pos) const {
std::vector<size_t> ind(this->shape.size(), 0);
auto pos_copy = pos;
for (int i = (int)this->shape.size() - 1; i >= 0; --i) {
ind[i] = pos_copy % this->shape[i];
pos_copy /= this->shape[i];
}
return ind;
}


void reshape(const std::vector<int> &shape);


template <typename... Args>
void reshape(const std::vector<int> &shape, const size_t dim,
const Args... args) {
std::vector<int> shape_copy = shape;
shape_copy.push_back(dim);
reshape(shape_copy, args...);
return;
}


template <typename... Args> void reshape(const int dim, const Args... args) {
std::vector<int> shape(1);
shape[0] = dim;
reshape(shape, args...);
return;
}



void diag_add(const Float alpha);


void diag_sub(const Float alpha);


void diag_mul(const Float alpha);


void diag_div(const Float alpha);


void diag_add(const vector<Float> &vec);
void diag_add(const view1D<vector<Float>, Float> &vec);
void diag_add(const view1D<matrix::Dense<Float>, Float> &vec);
void diag_add(const view1D<tensor::tensor_Dense<Float>, Float> &vec);


void diag_sub(const vector<Float> &vec);
void diag_sub(const view1D<vector<Float>, Float> &vec);
void diag_sub(const view1D<matrix::Dense<Float>, Float> &vec);
void diag_sub(const view1D<tensor::tensor_Dense<Float>, Float> &vec);


void diag_mul(const vector<Float> &vec);
void diag_mul(const view1D<vector<Float>, Float> &vec);
void diag_mul(const view1D<matrix::Dense<Float>, Float> &vec);
void diag_mul(const view1D<tensor::tensor_Dense<Float>, Float> &vec);


void diag_div(const vector<Float> &vec);
void diag_div(const view1D<vector<Float>, Float> &vec);
void diag_div(const view1D<matrix::Dense<Float>, Float> &vec);
void diag_div(const view1D<tensor::tensor_Dense<Float>, Float> &vec);
};
} 
} 
