#pragma once
#include <fstream>

class Matrix
{
private:
int rows;
int columns;
public:
double* Data;

Matrix(int rows, int columns, double* data);

Matrix(int rows, int columns);

static Matrix GenerateMatrix(int n, int m);

void Clear();

double Get(int i, int j) const;

int GetRows() const;

int GetColumns() const;

Matrix Transpose() const;

Matrix operator* (const Matrix& m) const;

friend  std::ofstream& operator<< (std::ofstream& out, const Matrix& m);
};