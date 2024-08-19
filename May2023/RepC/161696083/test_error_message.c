#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
int errors, i;
int test_error_message() {
i = 0;
OMPVV_INFOMSG("If successful, test should print an \"error message success\" at the beginning of the test");
#pragma omp parallel
{
#pragma omp single
{
#pragma omp error severity(warning) message("error message success")
i+=5;
}
}
OMPVV_TEST_AND_SET_VERBOSE(errors, i != 5);
OMPVV_ERROR_IF(errors > 0, "Error directive caused execution error");
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_error_message() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
