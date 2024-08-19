#include <iostream>
#include <omp.h>
#include <cassert>
#include "ompvv.h"
#include <cmath>
#define N 1000
template<typename T>
class Mapper {
private:
T* ptr;
bool not_mapped;
public:
Mapper (T* p) : ptr(p) {
not_mapped = !omp_target_is_present(ptr, omp_get_default_device());
T* solutionPtr = ptr;
#pragma omp target enter data map(to:solutionPtr[0:1]) if(not_mapped)
}
~Mapper() {
T* solutionPtr = ptr;
#pragma omp target exit data map(delete: solutionPtr[0:1]) if(not_mapped)
ptr = NULL;
}
};
class B : public Mapper<B> {
protected:
int n;
private:
double* x;
double sumB; 
public:
B(int nn) : Mapper<B>(this), n(nn) {
x = new double[n];
std::fill(x, x+n, 0);
double* solutionX = x;
int &cpy_n = n;
#pragma omp target enter data map(to:solutionX[0:n], cpy_n)
}
void modifyB() {
double * cpy_x = x; 
int &cpy_n = n;
double &cpy_sumB = sumB;
#pragma omp target defaultmap(tofrom: scalar)
{
sumB = 0.0;
for (int i = 0; i < cpy_n; ++i) {
cpy_x[i] += 1.0;
sumB += cpy_x[i];
}
} 
}
void getValuesB(double& b_sum, double* b_array) {
double * cpy_x = x;
int &cpy_n = n;
double &cpy_sum = sumB;
#pragma omp target defaultmap(tofrom: scalar) map(from: b_array[0:n])
{
b_sum = cpy_sum;
for (int i = 0; i < cpy_n; i++)
b_array[i] = cpy_x[i];
}
}
};
class A : public Mapper<A>, public B {
private:
int sumA; 
int* y;
public:
A(int s) : Mapper<A>(this), B(s){ 
y = new int[n];
std::fill(y, y+n, 0);
int *solutionY = y;
int &cpy_n = n;
#pragma omp target update to(cpy_n)
#pragma omp target enter data map(to:solutionY[0:n])
}
void modifyA() {
modifyB();
int *cpy_y = y;
int &cpy_n = n;
int &cpy_sumA = sumA;
#pragma omp target defaultmap(tofrom: scalar)
{
cpy_sumA = 0;
for (int i = 0; i < cpy_n; ++i) {
cpy_y[i] += 1;
cpy_sumA += cpy_y[i];
}
}
}
void getValuesA(int &a_sum, int* a_array, double& b_sum, double* b_array) {
getValuesB(b_sum, b_array);
int* cpy_y = y;
int &cpy_n = n;
int &cpy_sumA = sumA;
#pragma omp target defaultmap(tofrom:scalar) map(tofrom: a_array[0:n])
{
a_sum = cpy_sumA;
for (int i = 0; i < cpy_n; i++) {
a_array[i] = cpy_y[i];
}
}
}
};
int test_complex_class() {
OMPVV_INFOMSG("test_complex_class");
int sumY = 0, errors = 0, check_sumY = 0;
double sumX = 0.0, check_sumX = 0.0;
double * h_array_x = new double[N];
int * h_array_y = new int[N];
A *obj = new A(N);
obj->modifyA();
obj->modifyA();
obj->modifyA();
obj->getValuesA(sumY, h_array_y, sumX, h_array_x);
for (int i = 0; i < N; ++i) {
check_sumY += h_array_y[i];
check_sumX += h_array_x[i];
}
delete obj;
delete[] h_array_x;
delete[] h_array_y;
OMPVV_TEST_AND_SET(errors, check_sumY != 3*N);
OMPVV_TEST_AND_SET(errors, sumY != 3*N);
OMPVV_TEST_AND_SET(errors, std::abs(check_sumX - 3*N) > 0.00001);
OMPVV_TEST_AND_SET(errors, std::abs(sumX - 3*N) > 0.00001);
OMPVV_ERROR_IF(errors != 0, "N = %d, sumX = %f, check_sumX = %f, sumY=%d, check_sumY = %d", N, sumX, check_sumX, sumY, check_sumY);
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_complex_class());
OMPVV_REPORT_AND_RETURN(errors)
}
