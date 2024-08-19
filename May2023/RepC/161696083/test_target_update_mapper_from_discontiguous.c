#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
typedef struct{
size_t len;
double *data;
} T;
int i;
int errors = 0;
#pragma omp declare mapper(S: T v) map(to: v, v.len, v.data[0:v.len])
int target_update_from_mapper() {
T s;
s.len = N;
s.data = (double *)calloc(N,sizeof(double));
#pragma omp target  
{ 
for (int i = 0; i < s.len; i++) {
s.data[i] = i;
}
}
#pragma omp target data map(mapper(S), to:s) 
{ 
#pragma omp target map(mapper(S), to:s)
for (int i = 0; i < s.len; i++) {       
s.data[i] += i ;
}
#pragma omp target update from(s.data[0:s.len:2]) 
} 
for (i =0; i < s.len; i++) {
if(i%2){
OMPVV_TEST_AND_SET(errors, s.data[i] != i);
}
else{
OMPVV_TEST_AND_SET(errors, s.data[i] != 2 * i);
}
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, target_update_from_mapper());
OMPVV_REPORT_AND_RETURN(errors);
}
