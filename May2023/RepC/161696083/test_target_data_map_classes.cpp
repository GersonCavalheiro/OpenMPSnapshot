#include <iostream>
#include <omp.h>
#include "ompvv.h"
using namespace std;
#define N 1000
class A {
public:
int *h_array;
int size;
int sum;
A(int *array, const int s) : h_array(array), size(s) { sum = 0; }
~A() { h_array = NULL; }
};
int test_map_tofrom_class_heap() {
OMPVV_INFOMSG("test_map_tofrom_class_heap");
int sum = 0, errors = 0;
int *array = new int[N];
A *obj = new A(array, N);
#pragma omp target data map(from: array[0:N]) map(tofrom: obj[0:1])
{
#pragma omp target
{
int *tmp_h_array = obj->h_array;
obj->h_array = array;
int tmp = 0;
for (int i = 0; i < N; ++i) {
obj->h_array[i] = 1;
tmp += 1;
}
obj->h_array = tmp_h_array;
obj->sum = tmp;
} 
} 
for (int i = 0; i < N; ++i)
sum += obj->h_array[i];
OMPVV_TEST_AND_SET_VERBOSE(errors, (N != sum) || (N != obj->sum));
delete obj;
delete[] array;
return errors;
}
int test_map_tofrom_class_stack() {
OMPVV_INFOMSG("test_map_tofrom_class_stack");
int sum = 0, errors = 0;
int array[N];
A obj(array, N);
#pragma omp target data map(from: array[0:N]) map(tofrom: obj)
{
#pragma omp target
{
int *tmp_h_array = obj.h_array;
obj.h_array = array;
int tmp = 0;
for (int i = 0; i < N; ++i) {
obj.h_array[i] = 1;
tmp += 1;
}
obj.h_array = tmp_h_array;
obj.sum = tmp;
} 
} 
for (int i = 0; i < N; ++i)
sum += obj.h_array[i];
OMPVV_TEST_AND_SET_VERBOSE(errors, (N != sum) || (N != obj.sum));
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_tofrom_class_heap());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_tofrom_class_stack());
OMPVV_REPORT_AND_RETURN(errors);
}
