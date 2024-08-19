#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
int test_scope(){
int errors = 0;
int test_int = 1;
#pragma omp parallel shared(test_int)
{
#pragma omp scope private(test_int)
{
test_int += 1;
}
}
OMPVV_TEST_AND_SET_VERBOSE(errors,test_int != 1);
OMPVV_INFOMSG_IF(test_int == 2, "test int was not private");
return errors;
}
int main(){
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_scope() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
