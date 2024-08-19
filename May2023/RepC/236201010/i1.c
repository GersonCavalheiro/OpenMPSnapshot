#include <stdio.h>
#include <omp.h>
#define NUM_STEPS 100000
void main ()    
{
int nthreads = omt_get_max_threads();
double pi, step = 1.0 / NUM_STEPS
#pragma omp parallel
{
int i; id; double x, sum;  
id = omp_get_thread_num();  
for (i=id, sum=0.0; i < NUM_STEPS; i+= nthreads){
x = (i+0.5)*step; sum+=4.0/(1.0+x*x); 
}
pi += sum * step;			  
}
printf ("Integration Pi = %.10f\n", pi);
}