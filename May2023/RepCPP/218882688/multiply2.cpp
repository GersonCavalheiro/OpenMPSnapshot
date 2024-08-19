#include "../Matrix.hpp"
#include <omp.h>

void multiply(const Matrix &m1, const Matrix &m2, Matrix &result) {
auto m1a = m1.getArray();
auto m2a = m2.getArray();
auto ra = result.getArray();

for (unsigned r = 0; r < result.numRows(); ++r) {
#pragma omp parallel for
for (unsigned c = 0; c < result.numCols(); ++c) {
double sum = 0;
for (unsigned i = 0; i < m1.numRows(); ++i) {
sum = sum + m1a[r * m1.numCols() + i] * m2a[i * m2.numCols() + c];
}
ra[r * result.numCols() + c] = sum;
}
}
}
