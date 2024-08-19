#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 100
int test_atomic_compare() {
OMPVV_INFOMSG("test_atomic_compare");
int arr[N];
int errors = 0;
int max = 0, pmax = 0;
for(int i = 0; i < N; i++){
arr[i] = rand()%1000;
}
for(int i = 0; i < N; i++){ 
if(arr[i] > max){
max = arr[i];
}
}
#pragma omp target parallel for map(pmax) shared(pmax)
for(int i = 0; i < N; i++){
#pragma omp atomic compare
if(arr[i] > pmax){
pmax = arr[i];
}
}
OMPVV_TEST_AND_SET(errors, pmax != max);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_compare());
OMPVV_REPORT_AND_RETURN(errors);
}
