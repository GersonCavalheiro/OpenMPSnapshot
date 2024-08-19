#define num_steps 2000000000 
#include <stdio.h>
int main(int argc, char** argv)
{
double pi = 0.0;
long int i;
double x, interval_width;
interval_width = 1.0/(double)num_steps;
#pragma omp parallel for reduction(+:pi) private(x)
for (i = 0; i < num_steps; i++) {
x = (i+ 0.5) * interval_width;
pi += 1.0 / (x*x + 1.0);
}
pi = pi * 4.0 * interval_width;
printf ("PI=%f\n", pi);
return 0;
}
