#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ompvv.h"
int n=10, B[10];
int *x;
void init(int **A) {
int i;
*A = (int *) malloc(n*sizeof(int));
if (NULL == *A){ 
OMPVV_ERROR("This Test Has Failed, disregard other messages, array A is not properly allocated");
exit(-1);
}
x = *A;
for (i = 0; i < n; i++){
x[i] = 10;
B[i] = 0;
}
#pragma omp target enter data map(to:x[:n])
}
int main () {
int is_offloading;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
int i, errors = 0;
int *A;
init(&A);
#pragma omp target map(to: n) map(tofrom: B)
{
for (i = 0; i < n; i++)
B[i] = x[i];
}
for (i = 0; i < n; i++)
if (B[i] != 10) {
errors += 1;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, (errors != 0));
OMPVV_REPORT_AND_RETURN(errors);
}
