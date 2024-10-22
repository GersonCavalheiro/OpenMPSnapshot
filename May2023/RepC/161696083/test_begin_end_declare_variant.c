#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int arr[N]; 
int errors;
int i = 0;
void add(int *arr){
for (int i = 0; i < N; i++){ 
arr[i] = i+1;
}
}
#pragma omp begin declare variant match(construct={parallel}) 
void add(int *arr){
#pragma omp for
for (int i = 0; i < N; i++){
arr[i] = i + 2;
} 
}
#pragma omp end declare variant
#pragma omp begin declare variant match(construct={target}) 
void add(int *arr){
#pragma omp for
for (int i = 0; i < N; i++){
arr[i] = i + 3;
}
}
#pragma omp end declare variant
int test_wrapper() { 
add(arr);
for (int i = 0; i < N; i++){
OMPVV_TEST_AND_SET_VERBOSE(errors, arr[i] != i+1);
}
OMPVV_ERROR_IF(errors > 0, "Base function is not working properly")
errors = 0;
#pragma omp parallel
{
add(arr);
}		
for (int i = 0; i < N; i++){
OMPVV_TEST_AND_SET_VERBOSE(errors, arr[i] != i+2);
} 
OMPVV_ERROR_IF(errors>0, "Parallel variant function is not working properly")
errors=0;
#pragma omp target map(tofrom: arr)
{
add(arr);
}		
for(int i=0; i<N; i++){
OMPVV_TEST_AND_SET_VERBOSE(errors, arr[i] != i+3);
} 
OMPVV_ERROR_IF(errors>0, "Target variant function is not working properly")
return errors;
}
int main () {
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_wrapper());
OMPVV_REPORT_AND_RETURN(errors);
}  
