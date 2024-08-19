

#pragma once
#include <string>

using namespace std;


template <typename T> class Matrix {
public:
virtual ~Matrix(){};

virtual void
from_matrix_market_filepath(string const &matrix_market_filepath) = 0;

virtual void clear() = 0;

virtual void print() = 0;

public:
Matrix() {}

private:
Matrix(const Matrix &other) = delete;
Matrix &operator=(const Matrix &other) = delete;
};
