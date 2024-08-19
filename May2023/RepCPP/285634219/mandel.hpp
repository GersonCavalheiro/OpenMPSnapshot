
#pragma once

#include <exception>
#include <iomanip>
#include <iostream>
#include "common.h"

constexpr int row_size = 1080;
constexpr int col_size = 1920;
constexpr int max_iterations = 100;
int repetitions;

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

typedef struct {
float real;
float imag;
} ComplexF;

struct MandelParameters {
int row_count_;
int col_count_;
int max_iterations_;

MandelParameters(int row_count, int col_count, int max_iterations)
: row_count_(row_count),
col_count_(col_count),
max_iterations_(max_iterations) { }

int row_count() const { return row_count_; }
int col_count() const { return col_count_; }
int max_iterations() const { return max_iterations_; } 

float ScaleRow(int i) const { return -1.5f + (i * (2.0f / row_count_)); }

float ScaleCol(int i) const { return -1.0f + (i * (2.0f / col_count_)); }

int Point(const ComplexF& c) const {
int count = 0;
ComplexF z = {0, 0};
for (int i = 0; i < max_iterations_; ++i) {
auto r = z.real;
auto im = z.imag;
if (((r * r) + (im * im)) >= 4.0f) {
break;
}
z.real = r*r-im*im + c.real;
z.imag = 2*r*im + c.imag;

count++;
}
return count;
}
};

class Mandel {
private:
MandelParameters p_;
int *data_;  

public:

Mandel(int row_count, int col_count, int max_iterations)
: p_(row_count, col_count, max_iterations) {
data_ = new int[ p_.row_count() * p_.col_count() ];
}

~Mandel() { delete[] data_; }

MandelParameters GetParameters() const { return p_; }

void Print() {
if (p_.row_count() > 128 || p_.col_count() > 128) {
std::cout << "No Print() output due to size too large" << std::endl;
return;
}
for (int i = 0; i < p_.row_count(); ++i) {
for (int j = 0; j < p_.col_count_; ++j) {
std::cout << std::setw(1) << ((GetValue(i,j) >= p_.max_iterations()) ? "x" : " ");
}
std::cout << std::endl;
}
}

int *data() const { return data_; }

void SetValue(int i, int j, float v) { data_[i * p_.col_count_ + j] = v; }

int GetValue(int i, int j) const { return data_[i * p_.col_count_ + j]; }

void Verify(Mandel &m) {
if ((m.p_.row_count() != p_.row_count_) || (m.p_.col_count() != p_.col_count_)) {
std::cout << "Fail verification - matrix size is different" << std::endl;
throw std::runtime_error("Verification failure");
}

int diff = 0;
for (int i = 0; i < p_.row_count(); ++i) {
for (int j = 0; j < p_.col_count(); ++j) {
if (m.GetValue(i,j) != GetValue(i,j)) 
diff++;
}
}

double tolerance = 0.05;
double ratio = (double)diff / (double)(p_.row_count() * p_.col_count());

#if _DEBUG
std::cout << "diff: " << diff << std::endl;
std::cout << "total count: " << p_.row_count() * p_.col_count() << std::endl;
#endif

if (ratio > tolerance) {
std::cout << "Fail verification - diff larger than tolerance"<< std::endl;
throw std::runtime_error("Verification failure");
}
#if _DEBUG
std::cout << "Pass verification" << std::endl;
#endif
}
};


class MandelSerial : public Mandel {
public:
MandelSerial(int row_count, int col_count, int max_iterations)
: Mandel(row_count, col_count, max_iterations) { }

void Evaluate() {
MandelParameters p = GetParameters();

for (int i = 0; i < p.row_count(); ++i) {
for (int j = 0; j < p.col_count(); ++j) {
ComplexF c= {p.ScaleRow(i), p.ScaleCol(j)};
SetValue(i, j, p.Point(c));
}
}
}
};

class MandelParallel : public Mandel {
public:
MandelParallel(int row_count, int col_count, int max_iterations)
: Mandel(row_count, col_count, max_iterations) { }

double Evaluate(queue &q) {
MandelParameters p = GetParameters();

const int rows = p.row_count();
const int cols = p.col_count();

const int data_size = rows * cols;
const size_t data_size_bytes = data_size * sizeof(int);

int *data_buf = malloc_device<int>(data_size, q);
q.memcpy(data_buf, data(), data_size_bytes).wait();

size_t block_x = (rows + THREADS_PER_BLOCK_X - 1)/THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_X;
size_t block_y = (cols + THREADS_PER_BLOCK_Y - 1)/THREADS_PER_BLOCK_Y * THREADS_PER_BLOCK_Y;

size_t global_work_size[] = {block_x, block_y};
size_t local_work_size[] = {THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y};

common::MyTimer t_ker;

q.submit([&](handler &h) {
h.parallel_for<class mandel_kernel>(
nd_range<2>(range<2>(global_work_size[0], global_work_size[1]), 
range<2>(local_work_size[0], local_work_size[1])), [=] (nd_item<2> item) {
int i = item.get_global_id(0);
int j = item.get_global_id(1);
if (i < rows && j < cols)
data_buf[i * cols + j] = p.Point({p.ScaleRow(i), p.ScaleCol(j)});
});
}).wait();

common::Duration kernel_time = t_ker.elapsed();

q.memcpy(data(), data_buf, data_size_bytes).wait();

free(data_buf, q);

return kernel_time.count();
}
};
