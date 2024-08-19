#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
typedef struct newvec {
size_t len;
double *data;
} newvec_t;
size_t i;
int errors;
#pragma omp declare mapper(newvec_t v) map(v, v.data[0:v.len])
int target_update_to_mapper() {
OMPVV_TEST_OFFLOADING;
newvec_t s;
int errors;
s.data = (double *)calloc(N,sizeof(double));
s.len = N;
#pragma omp target data map(tofrom: s)
{ 
for (i = 0; i < s.len; i++) {
s.data[i] = i;
}
#pragma omp target update to(s.data[0:s.len:2])
#pragma omp target 
for (i = 0; i < s.len; i++) {
s.data[i] += i;
}
} 
for (i =0; i < N; i++) { 
if(i%2){ 
OMPVV_TEST_AND_SET(errors, s.data[i] != i);
}
else{   
OMPVV_TEST_AND_SET(errors, s.data[i] != 2*i);
}
} 
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, target_update_to_mapper());
OMPVV_REPORT_AND_RETURN(errors);
}
