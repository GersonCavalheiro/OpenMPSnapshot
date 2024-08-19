
#ifndef CCGL_UTILS_ARRAY_H
#define CCGL_UTILS_ARRAY_H

#include <new> 
#include <cstdarg> 
#include <iostream>
#include <vector>

#include "basic.h"

using std::vector;
using std::cout;
using std::endl;
using std::nothrow;

namespace ccgl {

namespace utils_array {

template <typename T, typename INI_T>
bool Initialize1DArray(int row, T*& data, INI_T init_value);


template <typename T, typename INI_T>
bool Initialize1DArray(int row, T*& data, INI_T* init_data);

template <typename T, typename INI_T>
bool Initialize1DArray4ItpWeight(int row, T*& data, INI_T* init_data, int itp_weight_data_length);

template <typename T, typename INI_T>
bool Initialize2DArray(int row, int col, T**& data, INI_T init_value);


template <typename T, typename INI_T>
bool Initialize2DArray(int row, int col, T**& data, INI_T** init_data);


template <typename T1, typename T2>
bool Initialize2DArray(T1* init_data, int& rows, int& max_cols, T2**& data);


template <typename T>
void Release1DArray(T*& data);


template <typename T>
void Release2DArray(T**& data);


template <typename T>
void BatchRelease1DArray(T*& data, ...);


template <typename T>
void BatchRelease2DArray(int nrows, T**& data, ...);


void Output1DArrayToTxtFile(int n, const float* data, const char* filename);


void Output2DArrayToTxtFile(int rows, int cols, const float** data, const char* filename);


template <typename T>
void Read1DArrayFromTxtFile(const char* filename, int& rows, T*& data);


template <typename T>
void Read2DArrayFromTxtFile(const char* filename, int& rows, T**& data);


template <typename T>
void Read2DArrayFromString(const char* s, int& rows, T**& data);


template <typename T>
bool ValueInVector(T val, const vector<T>& vec);


template <typename T>
void RemoveValueInVector(T val, vector<T>& vec);


template <typename T>
class Array2D {
T** data_ptr;
vuint32_t m_rows;
vuint32_t m_cols;

T** create2DArray(vuint32_t nrows, vuint32_t ncols, const T& val = T()) {
T** ptr = nullptr;
T* pool = nullptr;
try {
ptr = new(nothrow) T*[nrows];  
pool = new(nothrow) T[nrows*ncols];  
for (vuint32_t i = 0; i < nrows * ncols; i++) {
pool[i] = val;
}
for (vuint32_t i = 0; i < nrows; ++i, pool += ncols) {
ptr[i] = pool;
}
return ptr;
} catch (std::bad_alloc& ex) {
delete[] ptr; 
}
}

public:
typedef T value_type;
T** data() {
return data_ptr;
}

vuint32_t get_rows() const { return m_rows; }

vuint32_t get_cols() const { return m_cols; }

Array2D() : data_ptr(nullptr), m_rows(0), m_cols(0) {}
Array2D(vuint32_t rows, vuint32_t cols, const T& val = T()) {
if (rows <= 0)
throw std::invalid_argument("number of rows is 0"); 
if (cols <= 0)
throw std::invalid_argument("number of columns is 0"); 
data_ptr = create2DArray(rows, cols, val);
m_rows = rows;
m_cols = cols;
}

~Array2D() {
if (data_ptr) {
delete[] data_ptr[0];  
delete[] data_ptr;     
}
}

Array2D(const Array2D& rhs) : m_rows(rhs.m_rows), m_cols(rhs.m_cols) {
data_ptr = create2DArray(m_rows, m_cols);
std::copy(&rhs.data_ptr[0][0], &rhs.data_ptr[m_rows - 1][m_cols], &data_ptr[0][0]);
}

Array2D(Array2D&& rhs) NOEXCEPT {
data_ptr = rhs.data_ptr;
m_rows = rhs.m_rows;
m_cols = rhs.m_cols;
rhs.data_ptr = nullptr;
}

Array2D& operator=(Array2D&& rhs) NOEXCEPT {
if (&rhs != this) {
swap(rhs, *this);
}
return *this;
}

void swap(Array2D& left, Array2D& right) {
std::swap(left.data_ptr, right.data_ptr);
std::swap(left.m_cols, right.m_cols);
std::swap(left.m_rows, right.m_rows);
}

Array2D& operator = (const Array2D& rhs) {
if (&rhs != this) {
Array2D temp(rhs);
swap(*this, temp);
}
return *this;
}

T* operator[](vuint32_t row) {
return data_ptr[row];
}

const T* operator[](vuint32_t row) const {
return data_ptr[row];
}

void create(vuint32_t rows, vuint32_t cols, const T& val = T()) {
*this = Array2D(rows, cols, val);
}
};



template <typename T, typename INI_T>
bool Initialize1DArray(const int row, T*& data, const INI_T init_value) {
if (nullptr != data) {
cout << "The input 1D array pointer is not nullptr, without initialized!" << endl;
return false;
}
if (row <= 0) {
cout << "The data length MUST greater than 0!" << endl;
data = nullptr;
return false;
}
data = new(nothrow)T[row];
if (nullptr == data) {
delete[] data;
cout << "Bad memory allocated during 1D array initialization!" << endl;
data = nullptr;
return false;
}
T init = static_cast<T>(init_value);
#pragma omp parallel for
for (int i = 0; i < row; i++) {
data[i] = init;
}
return true;
}

template <typename T, typename INI_T>
bool Initialize1DArray(const int row, T*& data, INI_T* const init_data) {
if (nullptr != data) {
cout << "The input 1D array pointer is not nullptr, without initialized!" << endl;
return false;
}
data = new(nothrow) T[row];
if (nullptr == data) {
delete[] data;
cout << "Bad memory allocated during 1D array initialization!" << endl;
return false;
}
if (nullptr == init_data) {
cout << "The input parameter init_data MUST NOT be nullptr!" << endl;
return false;
}
#pragma omp parallel for
for (int i = 0; i < row; i++) {
data[i] = static_cast<T>(init_data[i]);
}
return true;
}

template <typename T, typename INI_T>
bool Initialize2DArray(const int row, const int col, T**& data,
const INI_T init_value) {
if (nullptr != data) {
cout << "The input 2D array pointer is not nullptr, without initialized!" << endl;
return false;
}
data = new(nothrow) T*[row];
if (nullptr == data) {
delete[] data;
cout << "Bad memory allocated during initialize rows of the 2D array!" << endl;
return false;
}
T* pool = nullptr;
pool = new(nothrow) T[row * col];
if (nullptr == pool) {
delete[] pool;
cout << "Bad memory allocated during initialize data pool of the 2D array!" << endl;
return false;
}
T init = static_cast<T>(init_value);
#pragma omp parallel for
for (int i = 0; i < row * col; i++) {
pool[i] = init;
}
for (int i = 0; i < row; ++i, pool += col) {
data[i] = pool;
}
return true;
}

template <typename T, typename INI_T>
bool Initialize2DArray(const int row, const int col, T**& data,
INI_T** const init_data) {
bool flag = Initialize2DArray(row, col, data, init_data[0][0]);
if (!flag) { return false; }
#pragma omp parallel for
for (int i = 0; i < row; i++) {
for (int j = 0; j < col; j++) {
data[i][j] = static_cast<T>(init_data[i][j]);
}
}
return true;
}

template <typename T1, typename T2>
bool Initialize2DArray(T1* init_data, int& rows, int& max_cols, T2**& data) {
int idx = 0;
rows = CVT_INT(init_data[idx++]);
data = new(nothrow) T2* [rows];
if (nullptr == data) {
delete[] data;
cout << "Bad memory allocated during initialize rows of the 2D array!" << endl;
return false;
}
T2* pool = nullptr;
int* cols = new int[rows];
max_cols = -1;
for (int i = 0; i < rows; i++) {
cols[i] = CVT_INT(init_data[idx]);
idx += cols[i] + 1;
if (cols[i] > max_cols) { max_cols = cols[i]; }
}
int length = idx - 1;
Initialize1DArray(length, pool, init_data + 1);
int pos = 0;
for (int i = 0; i < rows; ++i) {
data[i] = pool + pos;
pos += cols[i] + 1;
}
delete[] cols;
return true;
}

template <typename T>
void Release1DArray(T*& data) {
if (nullptr != data) {
delete[] data;
data = nullptr;
}
}

template <typename T>
void Release2DArray(T**& data) {
if (nullptr == data) {
return;
}
delete[] data[0]; 
delete[] data; 
data = nullptr;
}

template <typename T>
void BatchRelease1DArray(T*& data, ...) {
va_list arg_ptr;
va_start(arg_ptr, data);
Release1DArray(data);
T* arg_value = va_arg(arg_ptr, T*);
while (nullptr != arg_value) {
Release1DArray(arg_value);
arg_value = va_arg(arg_ptr, T*);
}
va_end(arg_ptr);
}

template <typename T>
void BatchRelease2DArray(const int nrows, T**& data, ...) {
va_list arg_ptr;
va_start(arg_ptr, data);
Release2DArray(nrows, data);
T** arg_value = va_arg(arg_ptr, T**);
while (nullptr != arg_value) {
Release2DArray(nrows, arg_value);
arg_value = va_arg(arg_ptr, T**);
}
va_end(arg_ptr);
}

template <typename T>
bool ValueInVector(const T val, const vector<T>& vec) {
if (vec.empty()) {
return false;
}
if (find(vec.begin(), vec.end(), val) == vec.end()) {
return false;
}
return true;
}

template <typename T>
void RemoveValueInVector(const T val, vector<T>& vec) {
for (auto iter = vec.begin(); iter != vec.end();) {
if (*iter == val) {
iter = vec.erase(iter);
} else {
++iter;
}
}
}

} 
} 

#endif 
