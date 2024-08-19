#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include "rdtsc.h"
#define ARRAYLENGTH 100000
static unsigned long long start, end;
int main(int argc, char* argv[])
{
int i;
long int sum_parallel = 0, sum_serial = 0;
int a[ARRAYLENGTH];
srand(time(NULL));
for (i = 0; i < ARRAYLENGTH; i++) {
a[i] = rand() % 10; 
}
int num_threads= omp_get_thread_num();;
omp_set_num_threads(num_threads);
start = rdtsc();
#pragma omp parallel for reduction(+:sum_parallel)
for (i = 0; i < ARRAYLENGTH; i++) {
sum_parallel += a[i];
}
end = rdtsc();
printf("Parallel CPU time in ticks: \t\t%llu\n",(end - start));
start = rdtsc();
for (i = 0; i < ARRAYLENGTH; i++) {
sum_serial += a[i];
}
end = rdtsc();
printf("Serial CPU time in ticks: \t\t%llu\n",(end - start));
printf("The parallel sum of the array is: \t\t%ld\n", sum_parallel);
printf("The serial sum of the array is: \t\t%ld\n", sum_serial);
return 0;
}
