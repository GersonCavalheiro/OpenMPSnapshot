#pragma once

#include <istream>
#include <ostream>
#include <iomanip>

class Matrix {
unsigned num_rows;
unsigned num_cols;

double *data;

public:

unsigned numRows() const {
return num_rows;
}

unsigned numCols() const {
return num_cols;
}

double *getArray() const {
return data;
}

Matrix(unsigned num_rows, unsigned num_cols) {
this->num_rows = num_rows;
this->num_cols = num_cols;
data = new double[num_rows * num_cols]();
}

Matrix(std::istream &input) {
input >> num_rows;
failOnStreamError(input);
input >> num_cols;
failOnStreamError(input);
data = new double[num_rows * num_cols];
for (unsigned r = 0; r < num_rows; ++r) {
for (unsigned c = 0; c < num_cols; ++c) {
input >> data[r * num_cols + c];
failOnStreamError(input);
}
}
}

Matrix(const Matrix &other) {
num_rows = other.num_rows;
num_cols = other.num_cols;
data = new double[num_rows * num_cols];
for (unsigned r = 0; r < num_rows; ++r) {
for (unsigned c = 0; c < num_cols; ++c) {
data[r * num_cols + c] = other.data[r * num_cols + c];
}
}
}

bool operator==(const Matrix &other) {
for (unsigned r = 0; r < num_rows; ++r) {
for (unsigned c = 0; c < num_cols; ++c) {
if (data[r * num_cols + c] != other.data[r * num_cols + c]) {
return false;
}
}
}
return true;
}

void stream(std::ostream &output) {
output << num_rows << " ";
output << num_cols << "\n";
for (unsigned r = 0; r < num_rows; ++r) {
for (unsigned c = 0; c < num_cols; ++c) {
output << std::fixed << std::setw(21) << std::setprecision(10) << std::setfill(' ')
<< data[r * num_cols + c] << " ";
}
output << "\n";
}
}

virtual ~Matrix() {
delete data;
}

private:

void failOnStreamError(std::istream &input) {
if (!input) {
throw std::runtime_error("Input error");
}
}
};
