#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_case(){
int errors = 0;
int sum = 0;
int test_lst[N];
for (int i = 0; i<N; i++){
test_lst[i] = 1;
}
#pragma omp target map(iterator(it = 0:N), tofrom: test_lst[it])
{
for(int i = 0; i < N; i++){
test_lst[i] = 2;
}
}
for (int i = 0; i < N; i++){
sum += test_lst[i];
}
OMPVV_TEST_AND_SET(errors, (sum != 2*N));
return (errors);
}
int main(){
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors,test_case() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
