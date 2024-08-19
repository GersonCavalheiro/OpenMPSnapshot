#include <iostream>
#include <iomanip>
#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <mkl.h>
#include "Matrix.h"

inline int Matrix::idx(int i, int j) const {return size*i + j;} 

void Matrix::fill(const double val) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (auto i=0; i<size*size; i++)
mat[i] = val;
}

void Matrix::copy(const Matrix &matrix) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (auto i=0; i<size*size; i++) {
mat[i] = matrix.mat[i];
}
}

void Matrix::gen_rand(const double val_min, const double val_max, const MKL_INT seed) const {
VSLStreamStatePtr stream;
int stat;
MKL_INT brng = VSL_BRNG_MT19937;
int method   = VSL_RNG_METHOD_UNIFORM_STD_ACCURATE;

stat = vslNewStream(&stream, brng, seed);
assert(stat == VSL_ERROR_OK);
#ifdef _UNIQUE_PTR
stat = vdRngUniform(method, stream, size*size, mat.get(), val_min, val_max);
#else
stat = vdRngUniform(method, stream, size*size, mat, val_min, val_max);
#endif
assert(stat == VSL_ERROR_OK);
stat = vslDeleteStream(&stream);
assert(stat == VSL_ERROR_OK);
}

int Matrix::lu_decomp(const double *a, int *ipivot, double *lu) const {
int i, j, k;
int ip, tmp_ip;
double tmp, max0, w;
#ifdef _OPENMP	
#pragma omp parallel private(i)
{
#pragma omp for
#endif
for (i=0; i<size*size; i++)
lu[i] = a[i];
#ifdef _OPENMP
#pragma omp for
#endif
for (i=0; i<size; i++)
ipivot[i] = i;
#ifdef _OPENMP
} 
#endif

for (k=0; k<size-1; k++) {
max0 = std::abs(lu[idx(k, k)]);
ip   = k;
for (i=k+1; i<size; i++) {
tmp = std::abs(lu[idx(i, k)]);
if (tmp > max0) {
max0 = tmp;
ip   = i;
}
}
if (max0 <= tol) {
std::cout << "one of diagonal component is smaller than " << tol << '\n';
return -1;
}

if (ip != k) {
#ifdef _OPENMP
#pragma omp parallel for private(tmp)
#endif
for (j=k; j<size; j++) {
tmp            = lu[idx(ip, j)];
lu[idx(ip, j)] = lu[idx(k , j)];
lu[idx(k , j)] = tmp;
}
tmp_ip     = ipivot[ip];
ipivot[ip] = ipivot[k];
ipivot[k]  = tmp_ip;
#ifdef _OPENMP
#pragma omp parallel for private(tmp)
#endif
for (j=0; j<k; j++) {
tmp            = lu[idx(k,  j)];
lu[idx(k,  j)] = lu[idx(ip, j)];
lu[idx(ip, j)] = tmp;
}
}
#ifdef _OPENMP
#pragma omp parallel for private(w)
#endif
for (i=k+1; i<size; i++) {
w             = lu[idx(i, k)]/lu[idx(k, k)];
lu[idx(i, k)] = w;
for (j=k+1; j<size; j++)
lu[idx(i, j)] = lu[idx(i, j)] - w*lu[idx(k, j)];
}
}
return 0;
}

int Matrix::inverse(double *a, double *a_inv) const {
double lu[size*size];
int ipivot[size];
int i, j, k;
double unit_vec[size], y[size];
double tmp;

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (i=0; i<size; i++) {
ipivot[i] = 0;
for (j=0; j<size; j++) {
lu[idx(i, j)]     = 0.0;
a_inv[idx(i, j)]  = 0.0;
}
}
int ret = lu_decomp(a, ipivot, lu);
if (ret) {
std::cout << "LU decomposition failed.\n";
return ret;
}
for (k=0; k<size; k++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (i=0; i<size; i++)
unit_vec[i] = 0.0;
unit_vec[k] = 1.0;
y[0] = unit_vec[ipivot[0]];
for (i=1; i<size; i++) {
tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:tmp)
#endif
for (j=0; j<i; j++)
tmp = tmp + lu[idx(i, j)]*y[j];
y[i] = unit_vec[ipivot[i]] - tmp;
}

a_inv[idx(size-1, k)] = y[size-1]/lu[idx(size-1, size-1)];
for (i=size-2; i>=0; i--) {
tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:tmp)
#endif
for (j=i+1; j<size; j++)
tmp = tmp + lu[idx(i, j)]*a_inv[idx(j, k)];
a_inv[idx(i, k)] = (y[i] - tmp)/lu[idx(i, i)];
}
}
return 0;
}

Matrix::Matrix() : size(1) {
#ifdef _UNIQUE_PTR
mat = std::make_unique<double[]>(size*size);
#else
mat = new double[size*size];
#endif
fill(0.0);
}

Matrix::Matrix(const int size) : size(size) {
#ifdef _UNIQUE_PTR
mat = std::make_unique<double[]>(size*size);
#else
mat = new double[size*size];
#endif
fill(0.0);
}
Matrix::Matrix(const Matrix &matrix) : size(matrix.size) {
#ifdef _UNIQUE_PTR
mat = std::make_unique<double[]>(size*size);
#else
mat = new double[size*size];
#endif
copy(matrix);
}

Matrix::~Matrix() {
#ifndef _UNIQUE_PTR
if (mat) delete[] mat;
#endif
}

Matrix& Matrix::operator=(const Matrix &rhs) {
if (this != &rhs) {
if (size != rhs.size) {
size = rhs.size;
#ifdef _UNIQUE_PTR
mat = std::make_unique<double[]>(size*size);
#else
delete[] mat;
mat = new double[size*size];
#endif
}
copy(rhs);
}
return *this;
}

Matrix& Matrix::operator=(const double rhs) {
fill(rhs);
return *this;
}

double Matrix::operator[] (const int i) const {
return mat[i];
}

Matrix Matrix::operator+(const Matrix &rhs) const {
Matrix matrix(*this);
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (auto i=0; i<size*size; i++)
matrix.mat[i] += rhs.mat[i];
return matrix;
}

Matrix Matrix::operator+(const double rhs) const {
Matrix matrix(*this);
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (auto i=0; i<size*size; i++)
matrix.mat[i] += rhs;
return matrix;
}

Matrix Matrix::operator-(const Matrix &rhs) const {
Matrix matrix(*this);
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (auto i=0; i<size*size; i++)
matrix.mat[i] -= rhs.mat[i];
return matrix;
}

Matrix Matrix::operator-(const double rhs) const {
Matrix matrix(*this);
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (auto i=0; i<size*size; i++)
matrix.mat[i] -= rhs;
return matrix;
}

Matrix Matrix::operator*(const Matrix &rhs) const {
int i, j, k;
Matrix matrix(size);
#ifdef _OPENMP
#pragma omp parallel for private(i, j, k)
#endif
for (i=0; i<size; i++) {
for (k=0; k<size; k++) {
for (j=0; j<size; j++) {
matrix.mat[idx(i, j)] += mat[idx(i, k)]*rhs.mat[idx(k, j)];
}
}
}
return matrix;
}

Matrix Matrix::operator*(const double rhs) const {
Matrix matrix(*this);
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (auto i=0; i<size*size; i++)
matrix.mat[i] *= rhs;
return matrix;
}

Matrix Matrix::operator/(const Matrix &rhs) const {
Matrix rhs_inv(size), matrix(*this);
#ifdef _UNIQUE_PTR
auto ret = inverse(rhs.mat.get(), rhs_inv.mat.get());
#else
auto ret = inverse(rhs.mat, rhs_inv.mat);
#endif
if (ret) {
std::cout << "LU decomposition or inverse failed.\n";
return Matrix(0);
}
matrix *= rhs_inv;
return matrix;
}

Matrix Matrix::operator/(const double rhs) const {
Matrix matrix(*this);
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (auto i=0; i<size*size; i++)
matrix.mat[i] /= rhs;
return matrix;
}

bool Matrix::operator==(const Matrix &rhs) const {
double diff, max_err;
max_err = 0.0;
#ifdef _OPENMP
#pragma omp parallel for private(diff) reduction(max:max_err)
#endif
for (auto i=0; i<size*size; i++) {
diff    = std::abs(mat[i]-rhs.mat[i]);
max_err = std::max(max_err, diff);
}
if (max_err >= tol) {
std::cout << "maximum error is: " << max_err << '\n';
return false;
}
return true;
}

bool Matrix::operator==(const double rhs) const {
double diff, max_err;
max_err = 0.0;
#ifdef _OPENMP
#pragma omp parallel for private(diff) reduction(max:max_err)
#endif
for (auto i=0; i<size*size; i++) {
diff    = std::abs(mat[i]-rhs);
max_err = std::max(max_err, diff);
}
if (max_err >= tol) {
std::cout << "maximum error is: " << max_err << '\n';
return false;
}
return true;
}

bool Matrix::operator!=(const Matrix &rhs) const {
if (*this == rhs) {
return false;
} else {
return true;
}
}

bool Matrix::operator!=(const double rhs) const {
if (*this == rhs) {
return false;
} else {
return true;
}
}

Matrix& Matrix::operator+=(const Matrix &rhs) {
*this = *this + rhs;
return *this;
}

Matrix& Matrix::operator+=(const double rhs) {
*this = *this + rhs;
return *this;
}

Matrix& Matrix::operator-=(const Matrix &rhs) {
*this = *this - rhs;
return *this;
}

Matrix& Matrix::operator-=(const double rhs) {
*this = *this - rhs;
return *this;
}

Matrix& Matrix::operator*=(const Matrix &rhs) {
*this = *this * rhs;
return *this;
}

Matrix& Matrix::operator*=(const double rhs) {
*this = *this * rhs;
return *this;
}

Matrix& Matrix::operator/=(const Matrix &rhs) {
*this = *this / rhs;
return *this;
}

Matrix& Matrix::operator/=(const double rhs) {
*this = *this / rhs;
return *this;
}

void Matrix::set(const double val_min, const double val_max, const int seed) {
gen_rand(val_min, val_max, seed);
}

void Matrix::set(const int i, const int j, const double val) {
mat[idx(i, j)] = val;
}

void Matrix::show() const {
for (auto i=0; i<size; i++) {
for (auto j=0; j<size; j++) {
std::cout << std::setw(14) << std::setprecision(5) << std::scientific <<mat[idx(i, j)] << " ";
}
std::cout << std::endl;
}
}

Matrix Matrix::transpose() const {
Matrix tmp(*this);
Matrix matrix(size);
int i, j;
#ifdef _OPENMP
#pragma omp parallel for private(i, j)
#endif
for (i=0; i<size; i++) {
for (j=0; j<size; j++) {
matrix.mat[idx(i, j)] = tmp.mat[idx(j, i)];
}
}
return matrix;
}

Matrix Matrix::inverse() const {
Matrix inv(size);
#ifdef _UNIQUE_PTR
auto ret = inverse(mat.get(), inv.mat.get());
#else
auto ret = inverse(mat, inv.mat);
#endif
if (ret) {
std::cout << __func__ << "() failed.\n";
inv = -1.0;
}
return inv;
}

double Matrix::trace() const {
double tr = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:tr)
#endif
for (auto i=0; i<size; i++)
tr += mat[idx(i, i)];

return tr;
}
double Matrix::determinant() const {
double det = 1.0;
double lu[size*size];
int ipivot[size];
int i, j;
#ifdef _OPENMP
#pragma omp parallel for private(i, j)
#endif
for (i=0; i<size; i++) {
ipivot[i] = 0;
for (j=0; j<size; j++)
lu[idx(i, j)]     = 0.0;
}
#ifdef _UNIQUE_PTR
int ret = lu_decomp(mat.get(), ipivot, lu);
#else
int ret = lu_decomp(mat, ipivot, lu);
#endif
if (ret) {
std::cout << "LU decomposition failed.\n";
return 1.0*ret;
}

#ifdef _OPENMP
#pragma omp parallel for reduction(*:det)
#endif
for (auto i=0; i<size; i++)
det *= lu[idx(i, i)];

return det;
}
