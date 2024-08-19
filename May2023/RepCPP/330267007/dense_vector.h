

#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "matrix.h"

using namespace std;


template <typename T> class DenseVector : public Matrix<T> {
public:
DenseVector(){};
~DenseVector();

void from_matrix_market_filepath(string const &matrix_market_filepath);

void from_num_zeros(int num_zeros);

void from_dense_vector(DenseVector<T> *vector);

void clear();

void print();

bool approx_equals(DenseVector<T> *other_vector, float threshold);


int dimension_get() { return dimension; };
T *values_get() { return values; };

private:
int dimension = 0; 
T *values = nullptr;
};

template <typename T> DenseVector<T>::~DenseVector() { this->clear(); };

template <typename T>
void DenseVector<T>::from_matrix_market_filepath(
string const &matrix_market_filepath) {
ifstream file(matrix_market_filepath);

while (file.peek() == '%')
file.ignore(2048, '\n');

int num_col, num_lines;
file >> dimension >> num_col >> num_lines;
values = new T[dimension];
int cur_val_idx = 0;
for (int l = 0; l < num_lines; l++) {
T data;
int row_idx, col;
file >> row_idx >> col >> data;
values[row_idx - 1] = data;
while (cur_val_idx < dimension &&
cur_val_idx < row_idx - 1) { 
values[cur_val_idx++] = 0;
}
cur_val_idx = row_idx;
}
while (cur_val_idx < dimension) { 
values[cur_val_idx++] = 0;
}

file.close();
};

template <typename T> void DenseVector<T>::from_num_zeros(int num_zeros) {
dimension = num_zeros;
values = new T[num_zeros];
for (int i = 0; i < num_zeros; i++) {
values[i] = 0;
}
}

template <typename T>
void DenseVector<T>::from_dense_vector(DenseVector<T> *vector) {
dimension = vector->dimension;
values = new T[dimension];
for (int i = 0; i < dimension; i++) {
values[i] = vector->values_get()[i];
}
}

template <typename T> void DenseVector<T>::clear() {
delete[] values;
values = nullptr;
dimension = 0;
}

template <typename T>
bool DenseVector<T>::approx_equals(DenseVector<T> *other_vector, float threshold) {
if (!other_vector || dimension != other_vector->dimension_get()) {
return false;
}
for (int i = 0; i < dimension; i++) {
if (abs(other_vector->values_get()[i] - values[i] + 0.0) /
min(abs(other_vector->values_get()[i]), abs(values[i])) >
threshold) {
return false;
}
}
return true;
}

template <typename T> void DenseVector<T>::print() {
cout << "DenseVector" << endl;
cout << "  dimension: " << dimension << endl;
cout << "  values:    ";
for (int i = 0; i < dimension; i++) {
if (values[i] != 0)
cout << "(" << i << ", " << values[i] << ") ";
}
cout << endl;
return;
};