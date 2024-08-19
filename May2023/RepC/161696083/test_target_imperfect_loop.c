#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 10
#define M 16
int test_target_imperfect_loop() {
OMPVV_INFOMSG("test_target_imperfect_loop");
int data1[N], data2[N][M];
int errors = 0;
for( int i = 0; i < N; i++){
data1[i] = 0;
for(int j = 0; j < M; j++){
data2[i][j] = 0;
}
}
#pragma omp target map(tofrom: data1, data2)
{
#pragma omp parallel for collapse(2)
for( int i = 0; i < N; i++){
data1[i] += i;
for(int j = 0; j < M; j++){
data2[i][j] += i + j;
}
}
}
for( int i=0;i<N;i++){
OMPVV_TEST_AND_SET(errors,data1[i] < i);
OMPVV_TEST_AND_SET(errors,data1[i] > i * M);
for(int j=0;j<M;j++){
OMPVV_TEST_AND_SET(errors,data2[i][j] != (i+j));
}
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_imperfect_loop());
OMPVV_REPORT_AND_RETURN(errors);
}
