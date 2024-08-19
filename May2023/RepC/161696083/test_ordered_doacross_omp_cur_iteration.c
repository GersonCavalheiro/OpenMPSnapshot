#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int errors;
int arr[N];
int ordered_doacross(){
int a[N];
int b[N];
int c[N];
a[0] = 0;
b[0] = 0;
c[0] = 0;
#pragma omp parallel for ordered
for(int i = 1; i < N; i++){
a[i] = i;
#pragma omp ordered doacross(sink: i-1)
b[i] = a[i-1];
#pragma omp ordered doacross(source:omp_cur_iteration)
c[i] = a[i] + b[i];
}
for(int i = 1; i < N; i++){
OMPVV_TEST_AND_SET(errors, a[i] != i);
OMPVV_TEST_AND_SET(errors, b[i] != i-1)
OMPVV_TEST_AND_SET(errors, c[i] != (i-1) + i);
}
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, ordered_doacross() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
