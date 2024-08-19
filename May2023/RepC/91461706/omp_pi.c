#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define REFPI 3.1415926535897932384626433
long double pi(long long nsteps){
long long i;
long double step, sum = 0.0;
step = 1.0 / ((long double) nsteps);
#pragma omp parallel
{
long double x; 
#pragma omp for reduction(+:sum)
for(i = 0; i < nsteps; i++){
x = (i + 0.5) * step;
sum += 4.0 / (1.0 + x * x);
}
}
return step * sum;
}
int main(int argc, char* argv[]){ 
long long nsteps;
if(argc < 2){
printf("Required argument: number of steps to compute pi for\n");
exit(-1);
}
nsteps = atoll(argv[1]);
long double comp_pi = pi(nsteps);
long double error = fabs(comp_pi - REFPI);
printf("%lli steps; pi = %.25Lf; error = %.25Lf\n", nsteps, comp_pi, error);
}
