#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 10
int errors = 0;
void fun();
void host_function();
#pragma omp declare target to(fun) device_type(nohost)
#pragma omp declare target
int a[N], b[N], c[N];  
int i = 0, dev=-9;
#pragma omp end declare target
#pragma omp declare variant(host_function) match(device={kind(host)})
void fun(){
for (i = 0; i < N; i++) {
a[i] += 1;
b[i] += 2;
c[i] += 3;
}
} 
void host_function() {
for (i = 0; i < N; i++) {
a[i] += 5;
b[i] += 5;
c[i] += 5;
}
}
int test_declare_target_device_type_nohost() { 
fun(); 
#pragma omp target update to(a,b,c)
#pragma omp target  
{
fun(); 
dev = omp_get_device_num();
}
#pragma omp target update from (a,b,c,dev)
if (dev != omp_get_initial_device()) {
for (i = 0; i < N; i++) { 
if ( a[i] != 6 || b[i] != 7 || c[i] != 8 ) {
errors++;
}
}
} else {
OMPVV_WARNING("Default device is the host device. Test ran on the host.");
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
OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target_device_type_nohost());
OMPVV_REPORT_AND_RETURN(errors);
}  
