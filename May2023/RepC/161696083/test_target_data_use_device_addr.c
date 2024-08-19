#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
int main() {
int errors = 0;
int device_data = 14, host_data=0;
OMPVV_TEST_OFFLOADING;
#pragma omp target data map(to: device_data)
{
int *dev_ptr;
#pragma omp target data use_device_addr(device_data)
{
dev_ptr = &device_data;
}
#pragma omp target map(to:device_data) map(tofrom: errors) map(from: host_data) is_device_ptr(dev_ptr)
{
if(&device_data != dev_ptr) {
errors++;
}
} 
#pragma omp target map(from: host_data) is_device_ptr(dev_ptr)
{
host_data = *dev_ptr;
}
} 
OMPVV_TEST_AND_SET(errors, host_data != 14);
OMPVV_REPORT_AND_RETURN(errors);
}
