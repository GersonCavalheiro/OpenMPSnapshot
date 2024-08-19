#include <omp.h>
#include "ompvv.h"
#include <cmath>
#define N 1000
class A {
private:
int h_array[N];
int size;
public:
A(const int s) : size(s) {
for (int i = 0; i < N; i++) {
h_array[i] = 0;
}
}
void modifyExplicit() {
int * theArray = this->h_array;
int theSize = size;
#pragma omp target map(theArray[0:N]) map(theSize) 
{
for (int i = 0; i < theSize; ++i)
theArray[i] += 1;
} 
}
int* getArray() {
return &h_array[0];
}
};
class B {
public:
static double VAR;
B() {}
~B() {}
#pragma omp declare target 
static int staticMethod() {
return 1;
}
#pragma omp end declare target 
};
double B::VAR = 1.0;
int test_explicit() {
OMPVV_INFOMSG("Explicit mapping test");
int sum = 0, errors = 0;
A *obj = new A(N);
obj->modifyExplicit();
int* h_array = obj->getArray();
for (int i = 0; i < N; ++i)
sum += h_array[i];
OMPVV_TEST_AND_SET_VERBOSE(errors, N != sum);
delete obj;
return errors;
}
int test_static () {
OMPVV_INFOMSG("Testing accessing a static variable");
int errors = 0;
double res = 0.0;
#pragma omp target map(tofrom: res)
{
res = B::VAR;
} 
OMPVV_TEST_AND_SET_VERBOSE(errors, std::abs(res - 1.0) > 0.0001)
return errors;
}
int test_static_method () {
OMPVV_INFOMSG("Testing static methods on the device");
int errors = 0;
int value = 0;
#pragma omp target map(tofrom: value) 
{
value = B::staticMethod();
} 
OMPVV_TEST_AND_SET_VERBOSE(errors, std::abs(value - 1.0) > 0.0001);
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_static_method() != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors,  test_static() != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors,  test_explicit() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
