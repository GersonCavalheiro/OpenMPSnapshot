#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int errors, i;
int test_defaultmap_present() {
struct test_struct {
int s; 
int S[N]; 
}; 
int scalar_var; 
int A[N]; 
struct test_struct new_struct; 
int *ptr; 
errors = 0;
scalar_var = 1; 
A[0] = 0; A[50] = 50;
new_struct.s = 10; new_struct.S[0] = 10; new_struct.S[1] = 10;
ptr = &A[0]; 
ptr[50] = 50; ptr[51] = 51;
#pragma omp target enter data map(to: scalar_var, A, new_struct, ptr)
#pragma omp target map(tofrom: errors) defaultmap(present)
{     
if(scalar_var != 1){errors++;}
if(A[0] != 0){errors++;}
if(A[50] != 50 || A[51] != 51){errors++;}
if(new_struct.s != 10){errors++;}
if(new_struct.S[0] != 10 || new_struct.S[1] != 10){errors++;}
scalar_var = 7; 
A[0] = 70; A[50] = 150;
new_struct.s = 110; new_struct.S[0] = 110; new_struct.S[1] = 110;
ptr = &A[0]; 
ptr[50] = 150; ptr[51] = 151;
}
OMPVV_TEST_AND_SET(errors, scalar_var == 7);
OMPVV_TEST_AND_SET(errors, A[0] == 70 || A[50] == 150 || A[51] == 151);
OMPVV_TEST_AND_SET(errors, new_struct.s == 110 || new_struct.S[0] == 110 || new_struct.S[1] == 110);
#pragma omp target exit data map(delete: scalar_var, A, new_struct, ptr)
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_present() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}           
