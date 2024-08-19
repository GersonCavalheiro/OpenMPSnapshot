#pragma once
#ifdef _UNIQUE_PTR
#include <memory>
#endif
#include <mkl.h>

class Matrix {
const double tol = 1.0e-10;
#ifdef _UNIQUE_PTR
std::unique_ptr<double[]> mat = nullptr;
#else
double *mat = NULL;
#endif
int    size = 0;

inline int idx(int i, int j) const;
void fill(const double val) const;
void copy(const Matrix &matrix);
void gen_rand(const double val_min, const double val_max, const MKL_INT seed) const;
int lu_decomp(const double *a, int *ipivot, double *lu) const;
int inverse(double *a, double *a_inv) const;

public:
Matrix();
Matrix(const int size);
Matrix(const Matrix &matrix);
virtual ~Matrix();
Matrix& operator=(const Matrix &rhs);
Matrix& operator=(const double rhs);
double operator[](const int i) const;
Matrix operator+(const Matrix &rhs) const;
Matrix operator+(const double rhs) const;
Matrix operator-(const Matrix &rhs) const;
Matrix operator-(const double rhs) const;
Matrix operator*(const Matrix &rhs) const;
Matrix operator*(const double rhs) const;
Matrix operator/(const Matrix &rhs) const;
Matrix operator/(const double rhs) const;
bool operator==(const Matrix &rhs) const;
bool operator==(const double rhs) const;
bool operator!=(const Matrix &rhs) const;
bool operator!=(const double rhs) const;
Matrix& operator+=(const Matrix &rhs);
Matrix& operator+=(const double rhs);
Matrix& operator-=(const Matrix &rhs);
Matrix& operator-=(const double rhs);
Matrix& operator*=(const Matrix &rhs);
Matrix& operator*=(const double rhs);
Matrix& operator/=(const Matrix &rhs);
Matrix& operator/=(const double rhs);
void set(const double val_min, const double val_max, const int seed);
void set(const int i, const int j, const double val);
void show() const;
Matrix transpose() const;
Matrix inverse() const;
double trace() const;
double determinant() const;
};
