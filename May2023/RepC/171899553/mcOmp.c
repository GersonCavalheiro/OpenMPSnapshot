#define _REENTRANT
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "timing.h"
int hits;
int main(int argc, char*argv[])
{
unsigned int state1, state2;
int     i;
double  x, y, rns;
if(argc<3)
{
perror("\nUEnter the number of loops and threads you want\n");
exit(-1);
}
int loop = atoi(argv[1]);
int numThreads = atoi(argv[2]);
rns = 1.0/(double)RAND_MAX;
state1 = (unsigned int)times(NULL);
timing_start();
#pragma omp parallel private(x, y, state2) num_threads(numThreads) reduction(+:hits) shared(state1)
{
#pragma omp critical
state2 = rand_r(&state1);
#pragma omp for
for (i=0; i<loop; i++) {
x = (double)rand_r(&state2) * rns;
y = (double)rand_r(&state2) * rns;
if (x*x + y*y < 1) {
hits++;
}
}
}
timing_stop();
print_timing();
return 0;
}
