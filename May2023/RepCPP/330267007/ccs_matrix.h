

#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "matrix.h"

using namespace std;


template <typename T> class CCSMatrix : public Matrix<T> {
public:
CCSMatrix(){};
~CCSMatrix();

void from_matrix_market_filepath(string const &matrix_market_filepath);

void to_lower_triangular();

void fill_diag(T fill_num, bool fill_nonzero);

void clear();

void print();


int num_row_get() { return num_row; };
int num_col_get() { return num_col; };
int num_val_get() { return num_val; };
int *column_pointer_get() { return column_pointer; };
int *row_index_get() { return row_index; };
T *values_get() { return values; };

private:
int num_row = 0;
int num_col = 0;
int num_val = 0;
int *column_pointer = nullptr;
int *row_index = nullptr;
T *values = nullptr;
};

template <typename T>
void CCSMatrix<T>::from_matrix_market_filepath(
string const &matrix_market_filepath) {
ifstream file(matrix_market_filepath);

while (file.peek() == '%')
file.ignore(2048, '\n');

file >> num_row >> num_col >> num_val;

column_pointer = new int[num_col + 1]; 
row_index = new int[num_val];
values = new T[num_val];
int cur_col_idx = 0;
for (int l = 0; l < num_val; l++) {
T data;
int row, col;
file >> row >> col >> data;
values[l] = data;
row_index[l] = row - 1;
while (cur_col_idx < num_col &&
cur_col_idx < col) { 
column_pointer[cur_col_idx++] = l;
}
}
while (cur_col_idx < num_col + 1) { 
column_pointer[cur_col_idx++] = num_val;
}

file.close();
}

template <typename T> void CCSMatrix<T>::to_lower_triangular() {
num_val = 0;
int val_idx = 0;
for (int j = 0; j < num_col; j++) {
while (row_index[val_idx] < j) {
val_idx++;
}
column_pointer[j] = num_val;
while (val_idx < column_pointer[j + 1]) {
values[num_val] = values[val_idx];
row_index[num_val] = row_index[val_idx];
val_idx++;
num_val++;
}
}
column_pointer[num_col] = num_val;
}

template <typename T>
void CCSMatrix<T>::fill_diag(T fill_num, bool fill_nonzero) {
int new_num_val = num_val;
int val_idx = 0;
for (int j = 0; j < num_col; j++) {
while (row_index[val_idx] < j) {
val_idx++;
}
if (row_index[val_idx] > j) {
new_num_val++;
}
val_idx = column_pointer[j + 1];
}
int *new_row_index = new int[new_num_val];
T *new_values = new T[new_num_val];
val_idx = 0;
num_val = 0;
for (int j = 0; j < num_col; j++) {
column_pointer[j] = num_val;
while (row_index[val_idx] < j) {
new_values[num_val] = values[val_idx];
new_row_index[num_val] = row_index[val_idx];
val_idx++;
num_val++;
}
if (row_index[val_idx] == j && fill_nonzero) {
new_values[num_val] = fill_num;
new_row_index[num_val] = row_index[val_idx];
val_idx++;
num_val++;
} else if (row_index[val_idx] > j) {
new_values[num_val] = fill_num;
new_row_index[num_val] = j;
num_val++;
val_idx++;
}
while (val_idx < column_pointer[j + 1]) {
new_values[num_val] = values[val_idx];
new_row_index[num_val] = row_index[val_idx];
val_idx++;
num_val++;
}
}
column_pointer[num_col] = num_val;
delete[] row_index;
delete[] values;
row_index = new_row_index;
values = new_values;
}

template <typename T> CCSMatrix<T>::~CCSMatrix() { this->clear(); }

template <typename T> void CCSMatrix<T>::clear() {
num_row = 0;
num_col = 0;
num_val = 0;
delete[] column_pointer;
delete[] row_index;
delete[] values;
}

template <typename T> void CCSMatrix<T>::print() {
cout << "CCSMatrix" << endl;
cout << "  num_row:        " << num_row << endl;
cout << "  num_col:        " << num_col << endl;
cout << "  num_val:        " << num_val << endl;
cout << "  column pointer: ";
for (int i = 0; i < num_col + 1; i++) {
cout << column_pointer[i] << " ";
}
cout << endl;
cout << "  row index:      ";
for (int i = 0; i < num_val; i++) {
cout << row_index[i] << " ";
}
cout << endl;
cout << "  values:         ";
for (int i = 0; i < num_val; i++) {
if (values[i] != 0)
cout << "(" << i << ", " << values[i] << ") ";
}
cout << endl;
return;
}