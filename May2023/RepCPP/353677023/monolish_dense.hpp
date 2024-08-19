#pragma once
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace matrix {




template <typename Float> class Dense {
private:

size_t rowN;


size_t colN;




mutable bool gpu_status = false;

public:

std::shared_ptr<Float> val;


size_t val_nnz = 0;


std::size_t alloc_nnz = 0;


bool val_create_flag = false;

Dense() { val_create_flag = true; }


void convert(const COO<Float> &coo);


void convert(const Dense<Float> &dense);


Dense(const COO<Float> &coo) {
val_create_flag = true;
convert(coo);
}


Dense(const Dense<Float> &dense);


Dense(const Dense<Float> &dense, Float value);


Dense(const size_t M, const size_t N);


Dense(const size_t M, const size_t N, const Float *value);


Dense(const size_t M, const size_t N, const std::vector<Float> &value);


Dense(const size_t M, const size_t N, const vector<Float> &value);


Dense(const size_t M, const size_t N,
const std::initializer_list<Float> &list);


Dense(const size_t M, const size_t N, const Float min, const Float max);


Dense(const size_t M, const size_t N, const Float min, const Float max,
const std::uint32_t seed);


Dense(const size_t M, const size_t N, const Float value);


void set_ptr(const size_t M, const size_t N, const std::vector<Float> &value);


void set_ptr(const size_t M, const size_t N, const Float *value);


[[nodiscard]] size_t get_row() const { return rowN; }


[[nodiscard]] size_t get_col() const { return colN; }


[[nodiscard]] size_t get_nnz() const { return get_row() * get_col(); }


void set_row(const size_t N) { rowN = N; };


void set_col(const size_t M) { colN = M; };




[[nodiscard]] std::string type() const { return "Dense"; }


void transpose();


void transpose(const Dense &B);


[[nodiscard]] double get_data_size() const {
return get_nnz() * sizeof(Float) / 1.0e+9;
}


[[nodiscard]] Float at(const size_t i, const size_t j) const;


[[nodiscard]] Float at(const size_t i, const size_t j) {
return static_cast<const Dense *>(this)->at(i, j);
};


void insert(const size_t i, const size_t j, const Float Val);


void print_all(bool force_cpu = false) const;


void send() const;


void recv();


void nonfree_recv();


void device_free() const;


[[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }


~Dense() {
if (val_create_flag) {
if (get_device_mem_stat()) {
device_free();
}
}
}


[[nodiscard]] const Float *data() const { return val.get(); }


[[nodiscard]] Float *data() { return val.get(); }


void resize(size_t N, Float Val = 0) {
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

void move(const tensor::tensor_Dense<Float> &tensor_dense);

void move(const tensor::tensor_Dense<Float> &tensor_dense, int rowN,
int colN);


[[nodiscard]] const Float *begin() const { return data(); }


[[nodiscard]] Float *begin() { return data(); }


[[nodiscard]] const Float *end() const { return data() + get_nnz(); }


[[nodiscard]] Float *end() { return data() + get_nnz(); }


void diag(vector<Float> &vec) const;
void diag(view1D<vector<Float>, Float> &vec) const;
void diag(view1D<matrix::Dense<Float>, Float> &vec) const;
void diag(view1D<tensor::tensor_Dense<Float>, Float> &vec) const;


void row(const size_t r, vector<Float> &vec) const;
void row(const size_t r, view1D<vector<Float>, Float> &vec) const;
void row(const size_t r, view1D<matrix::Dense<Float>, Float> &vec) const;
void row(const size_t r,
view1D<tensor::tensor_Dense<Float>, Float> &vec) const;


void col(const size_t c, vector<Float> &vec) const;
void col(const size_t c, view1D<vector<Float>, Float> &vec) const;
void col(const size_t c, view1D<matrix::Dense<Float>, Float> &vec) const;
void col(const size_t c,
view1D<tensor::tensor_Dense<Float>, Float> &vec) const;



void fill(Float value);


void operator=(const Dense<Float> &mat);


[[nodiscard]] Float &operator[](size_t i) {
if (get_device_mem_stat()) {
throw std::runtime_error("Error, GPU vector cant use operator[]");
}
return data()[i];
}


[[nodiscard]] bool equal(const Dense<Float> &mat,
bool compare_cpu_and_device = false) const;


[[nodiscard]] bool operator==(const Dense<Float> &mat) const;


[[nodiscard]] bool operator!=(const Dense<Float> &mat) const;


void reshape(const size_t row, const size_t col);



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
