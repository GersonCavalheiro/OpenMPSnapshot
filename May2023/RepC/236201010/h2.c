#include <stdio.h>
#include <omp.h>
#define NUM_STEPS 100000
void main ()
{
int i;
double x, pi, step, sum = 0.0;
step = 1.0 / (double) NUM_STEPS;
for (i=0; i< NUM_STEPS; i++){
x = (i+0.5)*step;
sum = sum + 4.0/(1.0+x*x);
}
pi = step * sum;
printf ("Integration Pi = %.10f\n", pi);
}