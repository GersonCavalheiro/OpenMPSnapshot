#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 64
int i;
int errors = 0;
int target_update_to_discontiguous() {
double *result;
result = (double *)calloc(N,sizeof(double));
#pragma omp target data map(tofrom: result[0:N])
{
for (int i = 0; i < N; i++) {
result[i] += i;
}
#pragma omp target update to(result[0:N:2])
#pragma omp target map(alloc: result[0:N]) 
{
for (int i = 0; i < N; i++) {
result[i] += i;
}
}
}
for (i =0; i < N; i++) {
if(i%2){
OMPVV_TEST_AND_SET(errors, result[i] != i);
}
else{
OMPVV_TEST_AND_SET(errors, result[i] != 2*i);
}
} 
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, target_update_to_discontiguous());
OMPVV_REPORT_AND_RETURN(errors);
}
