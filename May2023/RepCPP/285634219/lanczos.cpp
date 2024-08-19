#include <cassert>
#include <iostream>
#include <unistd.h>
#include <cmath>
#include <utility>

#include "eigen.h"
#include "matrix.h"
#include "lanczos.h"
#include "cycle_timer.h"

#define THREADS_PER_BLOCK 256



#pragma omp declare target
template <typename T>
void multiply_inplace_kernel(const int n, T* x, const T k) {
#pragma omp target teams distribute parallel for thread_limit(THREADS_PER_BLOCK)
for (int i = 0; i < n; i++) x[i] *= k;
}
#pragma omp end declare target



#pragma omp declare target
template <typename T>
void saxpy_inplace_kernel(const int n, T* y, const T *x, const T a) {
#pragma omp target teams distribute parallel for thread_limit(THREADS_PER_BLOCK)
for (int i = 0; i < n; i++) y[i] += a * x[i];
}
#pragma omp end declare target


#pragma omp declare target
template <typename T>
T device_dot_product(const int n, const T *x, const T *y) {
T result = 0;
const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma omp target teams distribute parallel for reduction(+:result) \
num_teams(blocks) thread_limit(THREADS_PER_BLOCK)
for (int index = 0; index < n; index++) 
result += x[index] * y[index];

return result;
}
#pragma omp end declare target



template <typename T>
symm_tridiag_matrix<T> gpu_lanczos(const csr_matrix<T> &m,
const vector<T> &v, const int steps) {
symm_tridiag_matrix<T> result(steps + 1);

int rows = m.row_size();
int cols = m.col_size();
int nonzeros = m.nonzeros();
assert(rows == cols);
assert(cols == v.size());

const int *row_ptr = m.row_ptr_data();
const int *col_ind = m.col_ind_data();
const T *values = m.values_data();
T *x = const_cast<T*>(v.data());

T* y = (T*) malloc (sizeof(T)*cols);
T* x_prev = (T*) malloc (sizeof(T)*cols);

double start_time, end_time;
const int row_nonzeros = nonzeros / rows;
int group_size = row_nonzeros > 16 ? 32 : 16;
group_size = row_nonzeros > 8 ? group_size : 8;
group_size = row_nonzeros > 4 ? group_size : 4;
group_size = row_nonzeros > 2 ? group_size : 2;


#pragma omp target data map (to: row_ptr[0:rows+1], \
col_ind[0:nonzeros], \
values[0:nonzeros], \
x[0:cols]) \
map (alloc: y[0:cols], x_prev[0:cols])
{
start_time = cycle_timer::current_seconds();
for (int i = 0; i < steps; i++) {
const int groups_per_block = THREADS_PER_BLOCK / group_size;
const int multiply_blocks = (rows + groups_per_block - 1) / groups_per_block;
#pragma omp target teams num_teams(multiply_blocks) thread_limit(THREADS_PER_BLOCK)
{
T result[THREADS_PER_BLOCK];
#pragma omp parallel 
{
int lid = omp_get_thread_num();
int index = omp_get_team_num()*omp_get_num_threads() + lid;
int r = index / group_size;
int lane = index % group_size;

result[lid] = 0;
if (r < rows) {
int start = row_ptr[r];
int end = row_ptr[r + 1];
for (int i = start + lane; i < end; i+= group_size)
result[lid] += values[i] * x[col_ind[i]];

int half = group_size / 2;
while (half > 0) {
if (lane < half) result[lid] += result[lid + half];
half /= 2;
}
if (lane == 0) y[r] = result[lid];
}
}
}

T product = device_dot_product(rows, x, y);

result.alpha(i) = product;

saxpy_inplace_kernel<T>(rows, y, x, -product);

if (i > 0) {
saxpy_inplace_kernel<T>(rows, y, x_prev, -result.beta(i - 1));
}

std::swap(x, x_prev);

result.beta(i) = T(std::sqrt(device_dot_product(rows, y, y)));

multiply_inplace_kernel<T>(rows, y, 1 / result.beta(i));

std::swap(x, y);
}
end_time = cycle_timer::current_seconds();
}
std::cout << "GPU Lanczos iterations: " << steps << std::endl;
std::cout << "GPU Lanczos time: " << end_time - start_time << " sec" << std::endl;


result.resize(steps);
return result;
}


template <typename T>
vector<T> gpu_lanczos_eigen(const csr_matrix<T> &matrix, int k, int steps) {
int cols = matrix.col_size();
assert(cols > 0);
vector<T> v(cols, 0);
v[0] = 1;
symm_tridiag_matrix<T> tridiag = gpu_lanczos(matrix, v, steps);
return lanczos_no_spurious(tridiag, k);
}


template vector<float> gpu_lanczos_eigen(const csr_matrix<float> &matrix, int k, int steps);
template vector<double> gpu_lanczos_eigen(const csr_matrix<double> &matrix, int k, int steps);
