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

#define MM_BANNER "%%MatrixMarket"
#define MM_MAT "matrix"
#define MM_VEC "vector"
#define MM_FMT "coordinate"
#define MM_TYPE_REAL "real"
#define MM_TYPE_GENERAL "general"
#define MM_TYPE_SYMM "symmetric"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace tensor {
template <typename Float> class tensor_Dense;
}
namespace matrix {
template <typename Float> class Dense;
template <typename Float> class CRS;
template <typename Float> class LinearOperator;




template <typename Float> class COO {
private:

size_t rowN;


size_t colN;




mutable bool gpu_status = false;

public:

std::vector<int> row_index;


std::vector<int> col_index;


std::shared_ptr<Float> val;


size_t val_nnz = 0;


std::size_t alloc_nnz = 0;


bool val_create_flag = false;

COO()
: rowN(0), colN(0), gpu_status(false), row_index(), col_index(),
val_nnz(0) {
val_create_flag = true;
}


COO(const size_t M, const size_t N)
: rowN(M), colN(N), gpu_status(false), row_index(), col_index(),
val_nnz(0) {
val_create_flag = true;
}


COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
const int *col, const Float *value);


COO(const size_t M, const size_t N, const size_t NNZ,
const std::vector<int> &row, const std::vector<int> &col,
const std::vector<Float> &value) {
this = COO(M, N, NNZ, row.data(), col.data(), value.data());
}


COO(const size_t M, const size_t N, const size_t NNZ,
const std::vector<int> &row, const std::vector<int> &col,
const vector<Float> &value) {
assert(value.get_device_mem_stat() == false);
this = COO(M, N, NNZ, row.data(), col.data(), value.data());
}


COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
const int *col, const Float *value, const size_t origin);


COO(const size_t M, const size_t N, const size_t NNZ,
const std::vector<int> &row, const std::vector<int> &col,
const std::vector<Float> &value, const size_t origin) {
this = COO(M, N, NNZ, row.data(), col.data(), value.data(), origin);
}


COO(const matrix::COO<Float> &coo);


COO(const matrix::COO<Float> &coo, Float value);


void convert(const matrix::CRS<Float> &crs);


COO(const matrix::CRS<Float> &crs) {
val_create_flag = true;
convert(crs);
}


void convert(const matrix::Dense<Float> &dense);


COO(const matrix::Dense<Float> &dense) {
val_create_flag = true;
convert(dense);
}

void convert(const matrix::LinearOperator<Float> &linearoperator);

COO(const matrix::LinearOperator<Float> &linearoperator) {
val_create_flag = true;
convert(linearoperator);
}


void set_row(const size_t M) { rowN = M; };


void set_col(const size_t N) { colN = N; };




void send() const {
throw std::runtime_error("error, GPU util of COO format is not impl. ");
};


void recv() const {
throw std::runtime_error("error, GPU util of COO format is not impl. ");
};


void device_free() const {};


[[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }


~COO() {
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

row_index.resize(N);
col_index.resize(N);
} else {
throw std::runtime_error("Error, not create vector cant use resize");
}
}



void input_mm(const std::string filename);


COO(const std::string filename) { input_mm(filename); }


void output_mm(const std::string filename) const;


void print_all(bool force_cpu = false) const;


void print_all(const std::string filename) const;


[[nodiscard]] Float at(const size_t i, const size_t j) const;


[[nodiscard]] Float at(const size_t i, const size_t j) {
return static_cast<const COO *>(this)->at(i, j);
};


void set_ptr(const size_t rN, const size_t cN, const std::vector<int> &r,
const std::vector<int> &c, const std::vector<Float> &v);


void set_ptr(const size_t rN, const size_t cN, const std::vector<int> &r,
const std::vector<int> &c, const size_t vsize, const Float *v);


[[nodiscard]] size_t get_row() const { return rowN; }


[[nodiscard]] size_t get_col() const { return colN; }


[[nodiscard]] size_t get_nnz() const { return val_nnz; }


void fill(Float value);


[[nodiscard]] std::vector<int> &get_row_ptr() { return row_index; }


[[nodiscard]] std::vector<int> &get_col_ind() { return col_index; }


[[nodiscard]] std::vector<Float> get_val_ptr() {
std::vector<Float> val(val_nnz);
for (size_t i = 0; i < val_nnz; ++i) {
val[i] = data()[i];
}
return val;
}


[[nodiscard]] const std::vector<int> &get_row_ptr() const {
return row_index;
}


[[nodiscard]] const std::vector<int> &get_col_ind() const {
return col_index;
}


[[nodiscard]] const std::vector<Float> get_val_ptr() const {
std::vector<Float> val(val_nnz);
for (size_t i = 0; i < val_nnz; ++i) {
val[i] = data()[i];
}
return val;
}



void transpose();


void transpose(const COO &B);


double get_data_size() const {
return 3 * get_nnz() * sizeof(Float) / 1.0e+9;
}


[[nodiscard]] std::string type() const { return "COO"; }


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



void operator=(const COO<Float> &mat);


[[nodiscard]] Float &operator[](size_t i) {
if (get_device_mem_stat()) {
throw std::runtime_error("Error, GPU vector cant use operator[]");
}
return data()[i];
}


[[nodiscard]] bool equal(const COO<Float> &mat,
bool compare_cpu_and_device = false) const;


[[nodiscard]] bool operator==(const COO<Float> &mat) const;


[[nodiscard]] bool operator!=(const COO<Float> &mat) const;


void insert(const size_t m, const size_t n, const Float val);

private:
void _q_sort(int lo, int hi);

public:

void sort(bool merge);
};


} 
} 
