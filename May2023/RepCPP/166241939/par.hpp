#ifndef PARFUNC_69BCD083
#define PARFUNC_69BCD083

#include <omp.h>
#include <iostream>

using namespace std;

namespace par {

template<typename T>
T *init_vector(long long length, T value) {
T *vector = new T[length];
#pragma omp parallel for
for (long long i = 0; i < length; ++i)
vector[i] = value;
return vector;
}

template<typename T, typename InitF>
T *init_vector_l(long long length, InitF lambda) {
T *vector = new T[length];
#pragma omp parallel for
for (long long i = 0; i < length; ++i)
vector[i] = lambda(i);
return vector;
}


template<typename T, typename F>
void map_vector(T *vector, long long length, F lambda) {
#pragma omp parallel for
for (long long i = 0; i < length; ++i)
vector[i] = lambda(vector[i]);
}

template<typename T, typename F>
T reduce_vector(T *vector, long long length, F lambda, T neutral) {
T reduction = neutral;
#pragma omp parallel
{
T priv_part = neutral;

#pragma omp for
for (long long i = 0; i < length; ++i) {
priv_part = lambda(priv_part, vector[i]);
}

#pragma omp critical
{
reduction = lambda(reduction, priv_part);
}
}
return reduction;
}

template<typename T>
T *copy_vector(T *vector, long long length) {
T *copy = new T[length];
#pragma omp parallel for
for (long long i = 0; i < length; ++i)
copy[i] = vector[i];
return copy;
}

template<typename T>
void linear_transform_vector(T *vector, long long length, T scale, T step) {
#pragma omp parallel for
for (long long i = 0; i < length; ++i)
vector[i] = scale * vector[i] + step;
}

template<typename T>
T **init_matrix(T **matrix, const long long dims[], T &value) {
long long d1 = dims[0], d2 = dims[1];
#pragma omp parallel for collapse(2)
for (long long i = 0; i < d1; ++i)
for (long long j = 0; j < d2; ++j)
matrix[i][j] = value;
return matrix;
}

template<typename T>
T **init_matrix(const long long dims[], T *vector) {
long long d1 = dims[0], d2 = dims[1];
T **matrix = new T *[d1];
#pragma omp parallel for
for (long long i = 0; i < d1; ++i) {
matrix[i] = copy_vector(vector, d2);
}
return matrix;
}

template<typename T>
T **init_matrix(const long long dims[], T value) {
long long d1 = dims[0], d2 = dims[1];
T **matrix = new T *[d1];
#pragma omp parallel for
for (long long i = 0; i < d1; ++i) {
matrix[i] = new T[d2];
#pragma omp parallel for
for (long long j = 0; j < d2; ++j)
matrix[i][j] = value;
}
return matrix;
}

template<typename T>
T **copy_matrix(T **matrix, const long long dims[]) {
long long d1 = dims[0], d2 = dims[1];
T **copy = new T[d1];
#pragma omp parallel for
for (long long i = 0; i < d1; ++i) {
copy[i] = new T[d2];
for (long long j = 0; j < d2; ++j)
copy[i][j] = matrix[i][j];
}
return copy;
}

template<typename T, typename F>
T **map_matrix(T **matrix, const long long dims[], F lambda) {
long long d1 = dims[0], d2 = dims[1];
#pragma omp parallel for collapse(2)
for (long long i = 0; i < d1; ++i)
for (long long j = 0; j < d2; ++j)
matrix[i][j] = lambda(matrix[i][j]);

return matrix;
}

template<typename T>
void print_matrix(T **m, const long long dims[]) {
long long d1 = dims[0], d2 = dims[1];
for (long long i = 0; i < d1; ++i) {
i == 0 ? cout << "/ " : i == d1 - 1 ? cout << "\\ " : cout << "| ";
for (long long j = 0; j < d2; ++j) {
j == d2 - 1 ? cout << m[i][j] : cout << m[i][j] << ", ";
}
i == 0 ? cout << " \\" : i == d1 - 1 ? cout << " /" : cout << " |";
cout << endl;
}

}


template<typename T, typename F>
T **wave_matrix(T **matrix, long long N, F lambda) {
long long x, y;

for (long long i = 1; i < N; ++i) { 
#pragma omp parallel for private(x, y)
for (y = 1; y <= i; ++y) {
x = i + 1 - y;
matrix[x][y] = lambda(matrix[x - 1][y], matrix[x][y - 1], matrix[x][y]);
}
}

for (long long i = 2; i < N; ++i) {
#pragma omp parallel for private(x, y)
for (y = i; y < N; ++y) {
x = N - 1 + i - y;
matrix[x][y] = lambda(matrix[x - 1][y], matrix[x][y - 1], matrix[x][y]);
}
}
}

template<typename T>
void destroy_matrix(T **m, const long long dims[]) {
long long d1 = dims[0];

for (long long i = 0; i < d1; ++i)
delete[] m[i];
delete[] m;
}


template<typename T>
void inline swap_elems(T **m, long long i, long long j) {
T tmp = m[i][j];
m[i][j] = m[j][i];
m[j][i] = tmp;
}

template<typename T>
T **transpose_matrix(T **m, long long dims[]) {
long long d1 = dims[0], d2 = dims[1];
if (d1 == d2) {
T **transposed = init_matrix(dims, m[0][0]);

#pragma omp parallel for collapse(2)
for (long long i = 0; i < d1; ++i)
for (long long j = 0; j < d2; ++j)
transposed[i][j] = m[j][i];
return transposed;
} else { 
return m;
}
}

template<typename T>
void transpose_matrix_in_place(T **m, long long dim) {
#pragma omp parallel for schedule(guided)
for (long long i = 0; i < dim - 1; ++i)
#pragma omp parallel for firstprivate(i)
for (long long j = i; j < dim; ++j)
swap_elems(m, i, j);
}

template<typename P1, typename P2>
void fork(P1 program1, P2 program2) {
#pragma omp parallel sections
{
#pragma omp section
{
program1();
}
#pragma omp section
{
program2();
}
}
}



template<typename P, typename T, typename R>
T *expand_vector(T *vector, const long long *indices, long long ind_n, R *values, P ex_function) {
T *exp_vector = (T *) malloc(sizeof(T) * ind_n);

#pragma omp parallel for
for (long long i = 0; i < ind_n; ++i) {
exp_vector[i] = ex_function(vector[indices[i]], values[i]);
}
return exp_vector;
}
}

#endif
