#include <omp.h>
#include <chrono>
#include <valarray>

#include "MultiplyMatrixBase.h"

using namespace std;

class MultiplyMatrixParallel : public MultiplyMatrixBase
{
public:
virtual string GetLabel() { return "matrix_mul_omp"; }
virtual void Run()
{
auto n = vector.size();
auto result = valarray<int>(n);
#pragma omp parallel for shared(result, n, matrix, vector)
for (size_t i = 0; i < matrix.size(); i++)
{
auto j = i % n;
int matrix_item = matrix[i];
int vector_item = vector[j];
int *value = &result[i / n];
auto increment = vector_item * matrix_item;
#pragma omp atomic
*value += increment;
}
}
};