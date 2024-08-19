#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int errors;
int i;
int test_defaultmap_with_to() {
struct test_struct {
int s;
int S[N];
};
int scalar; 
int A[N]; 
struct test_struct new_struct; 
int *ptr; 
scalar = 1;
new_struct.s = 0;
for (i = 0; i < N; i++) {
A[i] = 0;
new_struct.S[i] = 0;
}
#pragma omp target defaultmap(to) 
{
scalar = 17;    
A[0] = 5; A[1] = 5; 
new_struct.s = 10; new_struct.S[0] = 10; new_struct.S[1] = 10; 
ptr = &A[0]; ptr[50] = 70; ptr[51] = 71; 
}
OMPVV_TEST_AND_SET_VERBOSE(errors, scalar != 1); 
OMPVV_TEST_AND_SET_VERBOSE(errors, A[0] != 0 || A[1] != 0); 
OMPVV_TEST_AND_SET_VERBOSE(errors, A[50] != 0 || A[51] != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors, new_struct.s != 0 || new_struct.S[0] != 0 || new_struct.S[1] != 0);
return errors;
}
int test_defaultmap_with_from() {
struct test_struct {
int s;
int S[N];
};
int scalar; 
int A[N]; 
struct test_struct new_struct; 
int *ptr; 
scalar = 1;
new_struct.s = 0;
for (i = 0; i < N; i++) {
A[i] = 0;
new_struct.S[i] = 0;
}
#pragma omp target defaultmap(from) 
{
scalar = 17;    
A[0] = 5; A[1] = 5; 
new_struct.s = 10; new_struct.S[0] = 10; new_struct.S[1] = 10; 
ptr = &A[0]; ptr[50] = 50; ptr[51] = 51; 
}
OMPVV_TEST_AND_SET_VERBOSE(errors, scalar != 17); 
OMPVV_TEST_AND_SET_VERBOSE(errors, A[0] != 5 || A[1] != 5); 
OMPVV_TEST_AND_SET_VERBOSE(errors, A[50] != 50 || A[51] != 51);
OMPVV_TEST_AND_SET_VERBOSE(errors, new_struct.s != 10 || new_struct.S[0] != 10 || new_struct.S[1] != 10);
return errors;
}
int test_defaultmap_with_tofrom() {
struct test_struct {
int s;
int S[N];
};
int scalar; 
int A[N]; 
struct test_struct new_struct; 
int *ptr; 
scalar = 1;
new_struct.s = 0;
for (i = 0; i < N; i++) {
A[i] = 0;
new_struct.S[i] = 0;
}
#pragma omp target defaultmap(tofrom) 
{
scalar = 17;    
A[0] = 5; A[1] = 5; 
new_struct.s = 10; new_struct.S[0] = 10; new_struct.S[1] = 10; 
ptr = &A[0]; ptr[50] = 50; ptr[51] = 51; 
}
OMPVV_TEST_AND_SET_VERBOSE(errors, scalar != 17); 
OMPVV_TEST_AND_SET_VERBOSE(errors, A[0] != 5 || A[1] != 5); 
OMPVV_TEST_AND_SET_VERBOSE(errors, A[50] != 50 || A[51] != 51);
OMPVV_TEST_AND_SET_VERBOSE(errors, new_struct.s != 10 || new_struct.S[0] != 10 || new_struct.S[1] != 10);
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_with_to() != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_with_from() != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_with_tofrom() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
