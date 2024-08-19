#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#define N 100
int a[N];
int b[N];
int c[N];
int count, toggle=0;
int init_b(){
if(toggle % 2){
int i;
for (i = 0; i < N; i++) {
b[i] = b[i] * 2; 
}
toggle++;
return 1;
}
else{
toggle++;
return 0;
}
}
int main() {
int errors[2]={0,0}, i = 0, report_errors = 0, change_flag = 0;
for (i = 0; i < N; i++) {
a[i] = 10;
}
int is_offloading;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading); 
if (!is_offloading)
OMPVV_WARNING("It is not possible to test conditional data transfers "
"if the environment is shared or offloading is off. Not testing "
"anything");
for(count = 0; count < 4; count++){
for (i = 0; i < N; i++) {
b[i] = 2; 
c[i] = 0;
}
#pragma omp target data map(to: a[:N], b[:N]) map(tofrom: c)
{
#pragma omp target 
{
int j = 0;
for (j = 0; j < N; j++) {
c[j] = (a[j] + b[j]);
}
} 
change_flag = init_b();
#pragma omp target update if (change_flag) to(b[:N]) 
#pragma omp target 
{
int j = 0;
for (j = 0; j < N; j++) {
c[j] = (c[j] + b[j]);
}
} 
}
if (change_flag) {
for (i = 0; i < N; i++) {
if (c[i] != 16) {
errors[0] += 1;
}
}
}
else {
for (i = 0; i < N; i++) {
if (c[i] != 14) {
errors[1] += 1;
}
}
}
}
OMPVV_TEST_AND_SET_VERBOSE(report_errors, errors[0] > 0);
OMPVV_INFOMSG_IF(errors[0] > 0, "Target update test when if clause is true failed");
OMPVV_TEST_AND_SET_VERBOSE(report_errors, errors[1] > 0);
OMPVV_INFOMSG_IF(errors[1] > 0,  "Target update test when if clause is false failed");
OMPVV_REPORT_AND_RETURN(report_errors);
}
