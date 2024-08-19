#ifndef EXPECTED_POLICY
#define EXPECTED_POLICY DEFAULT
#endif
#include <omp.h>
#include <stdio.h>
#include <ctype.h>  
#include <stdlib.h>
#include <strings.h>  
#include "ompvv.h"
#define N 1028
typedef enum offload_policy
{MANDATORY, DISABLED, DEFAULT, UNKNOWN, NOTSET} offload_policy_t;
offload_policy_t get_offload_policy() {
char *env, *end;
size_t n;
env = getenv("OMP_TARGET_OFFLOAD");
if(env == NULL) return NOTSET;
end = env + strlen(env);
while (      *env && isspace(*(env  )) ) env++;
while (end != env && isspace(*(end-1)) ) end--;
n = (int)(end - env);
if      (n == 9 && !strncasecmp(env, "MANDATORY",n)) return MANDATORY;
else if (n == 8 && !strncasecmp(env, "DISABLED" ,n)) return DISABLED ;
else if (n == 7 && !strncasecmp(env, "DEFAULT"  ,n)) return DEFAULT  ;
else                                                 return UNKNOWN  ;
}
int main() {
int i, errors, isOffloading;
int on_init_dev;
int scalar;
int x[N];
errors = 0;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
offload_policy_t policy = get_offload_policy();
scalar = 17;
on_init_dev = 1;
for (i = 0; i < 0; i++) {
x[i] = 5;
}
#pragma omp target map(tofrom: on_init_dev, scalar, x) 
{
on_init_dev=omp_is_initial_device();
scalar = scalar + 53;
for (i = 0; i < 0; i++) {
x[i] = i*2;
}
}
OMPVV_TEST_AND_SET_VERBOSE(errors, scalar != 70);   
for (i = 0; i < 0; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != i*2);
}
OMPVV_ERROR_IF(policy==DEFAULT && isOffloading == 1 && on_init_dev != 0, "Did not follow DEFAULT policy and executed target region on the host even though a device was available");
OMPVV_TEST_AND_SET(errors, policy==DEFAULT && isOffloading == 1 && on_init_dev != 0);
OMPVV_ERROR_IF(policy==DEFAULT && isOffloading == 0 && on_init_dev != 1, "Did not follow DEFAULT policy and executed target region on device even though offloading appears unavailable");
OMPVV_TEST_AND_SET(errors, policy==DEFAULT && isOffloading == 0 && on_init_dev != 1);
OMPVV_ERROR_IF(policy==DISABLED && on_init_dev == 0, "Did not follow DISABLED policy and executed target region on device instead of executing on host");
OMPVV_TEST_AND_SET(errors, policy==DISABLED && on_init_dev == 0);
OMPVV_ERROR_IF(policy==MANDATORY && isOffloading == 1 && on_init_dev != 0, "Did not follow MANDATORY, instead executed target region on host even though device was available");
OMPVV_TEST_AND_SET(errors, policy==MANDATORY && isOffloading == 1 && on_init_dev != 0);
OMPVV_ERROR_IF(policy==NOTSET, "OMP_TARGET_OFFLOAD has not been set");
OMPVV_ERROR_IF(policy==UNKNOWN,"OMP_TARGET_OFFLOAD has an unknown value '%s'", getenv("OMP_TARGET_OFFLOAD"));
OMPVV_ERROR_IF(policy!=NOTSET && policy!=EXPECTED_POLICY, "OMP_TARGET_OFFLOAD has unexpected value '%s'", getenv("OMP_TARGET_OFFLOAD"));
OMPVV_REPORT_AND_RETURN(errors);
}
