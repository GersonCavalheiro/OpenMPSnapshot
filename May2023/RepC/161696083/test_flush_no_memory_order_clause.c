#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
int errors = 0;
int main() {
int x = 0, y = 0;
#pragma omp parallel num_threads(2)
{
int thrd = omp_get_thread_num();
if (thrd == 0) {
x = 10;
#pragma omp flush
#pragma omp atomic write
y = 1;
} else {
int tmp = 0;
while (tmp == 0) {
#pragma omp atomic read
tmp = y;
}
#pragma omp flush
OMPVV_TEST_AND_SET_VERBOSE(errors, x != 10);
}
}
OMPVV_REPORT_AND_RETURN(errors);
}
