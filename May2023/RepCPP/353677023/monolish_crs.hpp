#pragma once
#include <exception>
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

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace tensor {
template <typename Float> class tensor_Dense;
template <typename Float> class tensor_COO;
} 
namespace matrix {
template <typename Float> class Dense;
template <typename Float> class COO;




template <typename Float> class CRS {
private:

size_t rowN;


size_t colN;




mutable bool gpu_status = false;


size_t structure_hash;

public:

std::shared_ptr<Float> val;


size_t val_nnz = 0;


std::size_t alloc_nnz = 0;


bool val_create_flag = false;


std::vector<int> col_ind;


std::vector<int> row_ptr;

CRS() { val_create_flag = true; }


CRS(const size_t M, const size_t N, const size_t NNZ);


CRS(const size_t M, const size_t N, const size_t NNZ, const int *rowptr,
const int *colind, const Float *value);


CRS(const size_t M, const size_t N, const size_t NNZ, const int *rowptr,
const int *colind, const Float *value, const size_t origin);


CRS(const size_t M, const size_t N, const std::vector<int> &rowptr,
const std::vector<int> &colind, const std::vector<Float> &value);


CRS(const size_t M, const size_t N, const std::vector<int> &rowptr,
const std::vector<int> &colind, const vector<Float> &value);


void convert(COO<Float> &coo);


void convert(CRS<Float> &crs);


CRS(COO<Float> &coo) {
val_create_flag = true;
convert(coo);
}


CRS(const CRS<Float> &mat);


CRS(const CRS<Float> &mat, Float value);


void set_ptr(const size_t M, const size_t N, const std::vector<int> &rowptr,
const std::vector<int> &colind, const std::vector<Float> &value);


void set_ptr(const size_t M, const size_t N, const std::vector<int> &rowptr,
const std::vector<int> &colind, const size_t vsize,
const Float *value);


void print_all(bool force_cpu = false) const;


[[nodiscard]] size_t get_row() const { return rowN; }


[[nodiscard]] size_t get_col() const { return colN; }


[[nodiscard]] size_t get_nnz() const { return val_nnz; }


[[nodiscard]] std::string type() const { return "CRS"; }


void compute_hash();


[[nodiscard]] size_t get_hash() const { return structure_hash; }


void send() const;


void recv();


void nonfree_recv();


void device_free() const;


[[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }


~CRS() {
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


void transpose();


void transpose(const CRS &B);


[[nodiscard]] double get_data_size() const {
return (get_nnz() * sizeof(Float) + (get_row() + 1) * sizeof(int) +
get_nnz() * sizeof(int)) /
1.0e+9;
}


void fill(Float value);


void operator=(const CRS<Float> &mat);


[[nodiscard]] Float &operator[](size_t i) {
if (get_device_mem_stat()) {
throw std::runtime_error("Error, GPU vector cant use operator[]");
}
return data()[i];
}


[[nodiscard]] bool equal(const CRS<Float> &mat,
bool compare_cpu_and_device = false) const;


[[nodiscard]] bool operator==(const CRS<Float> &mat) const;


[[nodiscard]] bool operator!=(const CRS<Float> &mat) const;
};


} 
} 
