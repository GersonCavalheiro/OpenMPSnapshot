#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int errors = 0;
int test_wrapper() { 
#pragma omp parallel for
for (int i=1; i<N; i++){
int x,y,err = 0;
#pragma omp task depend(inout: x) shared(x) 
x=i;
#pragma omp task depend(inout: y) shared(y) 
y=i;
#pragma omp taskwait depend(in: x) 
OMPVV_TEST_AND_SET(err, x!= i);
#pragma omp taskwait depend(in: x,y) 
OMPVV_TEST_AND_SET(err, y!=i || x!=i);
#pragma omp atomic
errors += err;
}
return errors;
}
int main () {
OMPVV_TEST_AND_SET_VERBOSE(errors, test_wrapper());
OMPVV_REPORT_AND_RETURN(errors);
}  
