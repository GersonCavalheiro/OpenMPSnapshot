#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <thread>
using namespace std;

#include "jthread.hpp"

template<typename T, bool col_major=false>
class MatrixView{
public:
T* data_pointer;
const long nrow;
const long ncol;

MatrixView(T *data_pointer, const long nrow, const long ncol) : data_pointer(data_pointer), nrow(nrow),
ncol(ncol) {}

T &operator()(const int row, const int col) {
if (col_major) {
return data_pointer[row + col * nrow];
} else {
return data_pointer[col + row * ncol];
}
}

T operator()(const int row, const int col) const {
if (col_major) {
return data_pointer[row + col * nrow];
} else {
return data_pointer[col + row * ncol];
}
}
};


inline void get_matrix_to_matrix_dist(double *a, double *b, double *res, const long a_rows, const long b_rows, const long vec_dim){
const MatrixView<double, false> a_view(a, a_rows, vec_dim);
const MatrixView<double, false> b_view(b, b_rows, vec_dim);
int idx = 0;
double diff = 0.0;
for(auto i = 0; i != a_rows; i++){
for(auto j = 0; j != b_rows; j++){
double cur_sum = 0.0;
#pragma omp parallel for num_threads(4) reduction(+:cur_sum) private(diff)
for(auto k = 0; k != vec_dim; k++){
diff = a_view(i, k) - b_view(j, k);
cur_sum = cur_sum + diff * diff;
}
*(res + idx++) = sqrt(cur_sum);
}
}
}

inline void get_matrix_to_matrix_dist_multi_threading(double *a, double *b, double *res, const long a_rows, const long b_rows, const long vec_dim){
const int threads = omp_get_max_threads()-1;
jthread* thread_pool = new jthread[threads+1];
long start_ptr, size, local_rows;
start_ptr = 0l;
for(int i = 0; i<=threads; i++){
int start_idx_row = static_cast<int> (start_ptr / vec_dim);
if(a_rows * vec_dim - start_ptr < a_rows * vec_dim / threads){
local_rows = a_rows - start_idx_row;
}else{
local_rows = a_rows / threads;
}
size = local_rows * vec_dim;
thread_pool[i] = jthread(get_matrix_to_matrix_dist, &a[start_idx_row * vec_dim],
&b[0],
&res[start_idx_row * b_rows],
local_rows,
b_rows,
vec_dim);
start_ptr += size;
}
}

inline void get_pairwise_dist(double *a, const long a_rows, const long vec_dim, double *res){
const MatrixView<double, false> a_view(a, a_rows, vec_dim);
auto i = 0;
double diff = 0.0;
double cur_sum;
int index = 0;
for(; i != a_rows; i++){
for(auto j = 0; j != i; j++){
cur_sum = 0.0;
#pragma omp parallel for simd num_threads(2) reduction(+:cur_sum) private(diff)
for(auto k = 0; k < vec_dim; k++){
diff = (a_view(i, k) - a_view(j, k));
cur_sum = cur_sum + diff * diff;
}
*(res + index++) = *(res + (a_rows * j + i)) = sqrt(cur_sum);
}
}
}