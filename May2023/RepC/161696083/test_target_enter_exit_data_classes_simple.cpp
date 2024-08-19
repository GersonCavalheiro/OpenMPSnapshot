#include <iostream>
#include <omp.h>
#include "ompvv.h"
#define N 1000
class Simple {
private:
int *h_array;
int size;
int sum;
int *errors; 
public:
Simple(int s, int *err) : size(s) { 
sum = 0;
h_array = new int[size];
for (int i = 0; i < size; i ++)
h_array[i] = i;
errors = err;
int * helper_harray = this->h_array;
Simple * mySelf = this;
#pragma omp target enter data map(to: mySelf[0:1])
#pragma omp target enter data map(to: helper_harray[0:size])
}
~Simple() { 
int *helper_harray = this->h_array;
Simple * mySelf = this;
#pragma omp target exit data map(from: helper_harray[0:size])
for (int i = 0; i < N; ++i) {
OMPVV_TEST_AND_SET_VERBOSE(*errors, h_array[i] != 3*i);
}
#pragma omp target exit data map(from: mySelf[0:1])
OMPVV_TEST_AND_SET_VERBOSE(*errors, sum != 3*N*(N-1)/2);
delete[] h_array; 
}
void modify() {
int * helper_harray = this->h_array;
int &helper_sum = this->sum;
int &helper_size = this->size;
#pragma omp target defaultmap(tofrom:scalar)
{
helper_sum = 0;
for (int i = 0; i < helper_size; ++i) {
helper_harray[i] += i;
helper_sum += helper_harray[i];
}
}
}
void getDeviceAttributes(int * array_copy, int & sum_copy) {
int * helper_harray = this->h_array;
int &helper_sum = this->sum;
int &helper_size = this->size;
#pragma omp target map(from:array_copy[0:N], sum_copy) defaultmap(tofrom:scalar)
{
for (int i = 0; i < helper_size; ++i) {
array_copy[i] = helper_harray[i];
}
sum_copy = helper_sum;
}
}
};
int test_simple_class() {
OMPVV_INFOMSG("Testing simple class mapping");
int sum = 0, errors = 0, h_sum = 0;
int* h_array = new int[N];
Simple *obj = new Simple(N, &errors);
obj->modify();
obj->getDeviceAttributes(h_array, h_sum);
for (int i = 0; i < N; ++i) {
sum += h_array[i];
}
OMPVV_TEST_AND_SET_VERBOSE(errors, N*(N-1) != sum);
OMPVV_TEST_AND_SET_VERBOSE(errors, N*(N-1) != h_sum);
obj->modify();
delete obj;
delete[] h_array;
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET(errors, test_simple_class());
OMPVV_REPORT_AND_RETURN(errors);
}
