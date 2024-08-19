#ifndef MATRIX_H
#define MATRIX_H

#include <fstream>
#include <iostream>
#include <omp.h>

template<class T>
class Matrix {

public:

Matrix(); 

Matrix(char *filename); 

Matrix(size_t rows, size_t columns);

Matrix(const Matrix<T> &other); 

Matrix(T **data, size_t rows, size_t columns);

Matrix<T> &operator=(const Matrix<T> &other); 

Matrix<T> &operator+=(const Matrix<T> &other);

const Matrix<T> operator+(const Matrix<T> &other);

Matrix<T> &operator*=(const Matrix<T> &other);

const Matrix<T> operator*(const Matrix<T> &other);

Matrix<T> &operator*=(const T &multiplier);

T *&operator[](size_t index);

size_t get_rows() const;

size_t get_columns() const;

template<class U>
friend std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix);

template<class U>
friend std::istream &operator>>(std::istream &is, Matrix<U> &matrix);

~Matrix(); 

private:

size_t rows, columns;
T **data; 


void init_matrix(size_t rows, size_t columns);

void read_from_file(char *filename);

void write_to_file(char *filename);

void copy_data(T **other_data);
};


template<class T>
Matrix<T>::~Matrix() {

if (data) {
for (int i = 0; i < rows; ++i)
delete[] data[i];

delete[] data;
}

rows = columns = 0;
data = nullptr;

}


template<class T>
Matrix<T>::Matrix():rows(0), columns(0), data(nullptr) { }


template<class T>
Matrix<T>::Matrix(char *filename) : Matrix() {
read_from_file(filename);
}


template<class T>
Matrix<T>::Matrix(size_t rows, size_t columns) {

if (rows <= 0 || columns <= 0) {
std::cerr << "Invalid rows/columns -> rows: " << rows << ", columns: " << columns << std::endl;
return;
}

init_matrix(rows, columns);
}


template<class T>
Matrix<T>::Matrix(const Matrix<T> &other) : Matrix(other.rows, other.columns) {

if (other.data) {
copy_data(other.data);
}

}


template<class T>
size_t Matrix<T>::get_rows() const {
return rows;
}


template<class T>
size_t Matrix<T>::get_columns() const {
return columns;
}


template<class T>
Matrix<T>::Matrix(T **data, size_t rows, size_t columns): Matrix(rows, columns) {
if (data) {
copy_data(data);
}
}


template<class T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {

if (this != &other) {
this->~Matrix(); 

if (other.data)
new(this) Matrix(other); 
}

return *this;
}


template<class T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &other) {

if (this->columns == other.columns && this->rows == other.rows) {

for (int i = 0; i < other.columns; ++i) {
for (int j = 0; j < other.rows; ++j) {
this->data[j][i] += other.data[j][i];
}
}

} else {
std::cerr << "Can't add up matrices, due to the size difference" << std::endl;
}

return *this;
}


template<class T>
const Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) {

Matrix<T> result(*this); 

result += other;

return result;
}


template<class T>
Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &other) {

if (this->columns == other.rows) {

Matrix<T> result(this->rows, other.columns);

int	tid, nthreads, chunk = 0;

chunk = 1; 


#pragma omp parallel
{
tid = omp_get_thread_num(); 

if (tid == 0) {
nthreads = omp_get_num_threads(); 
printf("Starting multiply with %d threads\n", nthreads);
}


#pragma omp for schedule(static, chunk) 

for (int i = 0; i < this->rows; ++i) {

printf("Thread=%d did row=%d\n", tid, i);

for (int j = 0; j < other.columns; ++j) {

for (int k = 0; k < this->columns; ++k) {
result.data[i][j] += this->data[i][k] * other.data[k][j];
}
}
}
};

*this = result;
return *this;

} else {
std::cerr << "Can't multiply matrices: columns != rows" << std::endl;
}

return *this;
}


template<class T>
const Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) {

Matrix<T> result(*this); 

result *= other;

return result;
}


template<class T>
Matrix<T> &Matrix<T>::operator*=(const T &multiplier) {

for (int i = 0; i < columns; ++i) {
for (int j = 0; j < rows; ++j) {
data[j][i] *= multiplier;
}
}

return *this;
}


template<class T>
T *&Matrix<T>::operator[](size_t index) {

if (this->rows > index)
return this->data[index];

std::cerr << "Index out of bounds: " << index << ", max: " << (this->rows - 1) << std::endl;

throw std::runtime_error("Index out of bounds exception");
}


template<class U>
std::istream &operator>>(std::istream &is, Matrix<U> &matrix) {

is >> matrix.rows;
is >> matrix.columns;

matrix.init_matrix(matrix.rows, matrix.columns);

for (int i = 0; i < matrix.rows; ++i)
for (int j = 0; j < matrix.columns; ++j)
is >> matrix.data[i][j];

return is;
}


template<class U>
std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix) {

for (int i = 0; i < matrix.rows; ++i) {
for (int j = 0; j < matrix.columns; ++j) {
os << matrix.data[i][j] << "\t";
}
std::cout << std::endl;
}

return os;
}


template<class T>
void Matrix<T>::read_from_file(char *filename) {

if (!filename) {
std::cerr << "File shouldn't be null" << std::endl;
return;
}

std::ifstream is;
is.open(filename);

if (!is.is_open()) {
std::cerr << "Can't open file: " << filename << std::endl;
return;
}

operator>>(is, *this); 

is.close();
}


template<class T>
void Matrix<T>::write_to_file(char *filename) {
std::ofstream os;
os.open(filename);

if (!os.is_open()) {
std::cerr << "Can't open file: " << filename << std::endl;
return;
}

for (int i = 0; i < rows; ++i) {
for (int j = 0; j < columns; ++j) {
os << data[i][j] << '\t';
}
os << std::endl;
}

os.close();
}


template<class T>
void Matrix<T>::init_matrix(size_t rows, size_t columns) {

this->rows = rows;
this->columns = columns;

data = new T *[rows];

for (int i = 0; i < rows; ++i) {
data[i] = new T[columns];
}

for (int i = 0; i < columns; ++i) {
for (int j = 0; j < rows; ++j) {
data[j][i] = T();
}
}
}


template<class T>
void Matrix<T>::copy_data(T **other_data) {
if (other_data) {
for (int i = 0; i < columns; ++i) {
for (int j = 0; j < rows; ++j) {
this->data[j][i] = other_data[j][i];
}
}
}
}

#endif 
