#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int test_motion_present() {
int errors = 0;
struct test_struct {
int s; 
int S[N]; 
}; 
int scalar_var; 
int A[N]; 
struct test_struct new_struct; 
int *ptr; 
scalar_var = 1; 
A[0] = 0; A[50] = 50;
new_struct.s = 10; new_struct.S[0] = 10; new_struct.S[1] = 10;
ptr = &A[0]; 
ptr[50] = 50; ptr[51] = 51;
#pragma omp target update to(scalar_var, A, new_struct) 
if (omp_target_is_present(&scalar_var, omp_get_default_device()))
errors++;
if (omp_target_is_present(&A, omp_get_default_device()))
errors++;
if (omp_target_is_present(&new_struct, omp_get_default_device()))
errors++;
#pragma omp target enter data map(alloc: scalar_var, A, new_struct)
#pragma omp target update to(present: scalar_var, A, new_struct) 
if (!omp_target_is_present(&scalar_var, omp_get_default_device()))
errors++;
if (!omp_target_is_present(&A, omp_get_default_device()))
errors++;
if (!omp_target_is_present(&new_struct, omp_get_default_device()))
errors++;
#pragma omp target map(tofrom: errors) defaultmap(none) map(from: scalar_var, A, new_struct)
{     
if(scalar_var != 1){errors++;}
if(A[0] != 0 || A[50] != 50){errors++;}
if(A[50] != 50 || A[51] != 51){errors++;}
if(new_struct.s != 10){errors++;}
if(new_struct.S[0] != 10){errors++;}
}
#pragma omp target exit data map(release: scalar_var, A, new_struct)
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_motion_present() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}           
