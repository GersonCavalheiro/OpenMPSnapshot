#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 10
int errors = 0;
int a[N], b[N], c[N];  
int dev=-9, i = 0;
void target_function();
#pragma omp declare target enter(target_function) device_type(nohost)
#pragma omp declare target enter(a,b,c,i,dev)
#pragma omp declare variant(target_function) match(device={kind(nohost)})
void update() {
for (i = 0; i < N; i++) {
a[i] += 5;
b[i] += 5;
c[i] += 5;
}
}
void target_function(){
for (i = 0; i < N; i++) {
a[i] += 1;
b[i] += 2;
c[i] += 3;
}
} 
int test_declare_target_enter_device_type_nohost() { 
update();         
#pragma omp target update to(a,b,c)
#pragma omp target  
{
update();     
dev = omp_get_device_num();
}
#pragma omp target update from (a,b,c)
if (dev != omp_get_initial_device()) {
for (i = 0; i < N; i++) { 
if ( a[i] != 6 || b[i] != 7 || c[i] != 8 ) {
errors++;
}
}
} else {
OMPVV_WARNING("Default device is the host device. Target region was executed on the host");
for (i = 0; i < N; i++) { 
if ( a[i] != 10 || b[i] != 10 || c[i] != 10 ) {
errors++;
}
}
}
return errors;
}
int main () {
OMPVV_TEST_OFFLOADING;
for (i = 0; i < N; i++) {
a[i] = 0;
b[i] = 0;
c[i] = 0;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target_enter_device_type_nohost());
OMPVV_REPORT_AND_RETURN(errors);
}  
