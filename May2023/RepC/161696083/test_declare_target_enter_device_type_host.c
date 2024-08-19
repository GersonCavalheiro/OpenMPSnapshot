#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int errors = 0;
int a[N], b[N], c[N];  
int i = 0;
void update() { 
for (i = 0; i < N; i++) {
a[i] += 1;
b[i] += 2;
c[i] += 3;
}
}
#pragma omp declare target enter(a,b,c,i) 
#pragma omp declare target enter(update) device_type(host) 
int test_declare_target_device_type_host() { 
#pragma omp target update to(a,b,c)
#pragma omp target   
{
for (i = 0; i < N; i++) {
a[i] += i;
b[i] += 2 * i;
c[i] += 3 * i;
}
}
#pragma omp target update from( a, b, c)
update();
for (i = 0; i < N; i++) { 
if ( a[i] != i + 1 || b[i] != 2 * i + 2 || c[i] != 3 * i + 3 ) {
errors++;
}
}
return errors;
}
int main () {
for (i = 0; i < N; i++) {
a[i] = 0;
b[i] = 0;
c[i] = 0;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target_device_type_host());
OMPVV_REPORT_AND_RETURN(errors);
}  
