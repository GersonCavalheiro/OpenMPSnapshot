#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int errors = 0;
#pragma omp declare target 
int a[N], b[N], c[N];  
int i = 0;
#pragma omp end declare target
void update() { 
for (i = 0; i < N; i++) {
a[i] += 1;
b[i] += 2;
c[i] += 3;
}
}
#pragma omp declare target enter(update) device_type(any) 
int test_declare_target_enter_device_type_any() { 
#pragma omp target update to(a,b,c) 
#pragma omp target 
{
update();
}
#pragma omp target update from( a, b, c)
for (i = 0; i < N; i++) { 
if ( a[i] != i + 1 || b[i] != i + 3 || c[i] != i + 5) {
errors++;  
} 
}
update();
for (i = 0; i < N; i++) { 
if ( a[i] != i + 2 || b[i] != i + 5 || c[i] != i + 8) {
errors++;
}
}
return errors;
}
int main () {
for (i = 0; i < N; i++) {
a[i] = i;
b[i] = i + 1;
c[i] = i + 2;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target_enter_device_type_any());
OMPVV_REPORT_AND_RETURN(errors);
}  
