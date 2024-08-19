#include <stdio.h>
#include <omp.h>
#include "random.h"
static long num_trials = 10000;
int main ()
{
long i;  long Ncirc = 0;
double pi, x, y, test;
double r = 1.0;   
seed(-r, r);  
#pragma omp parallel for private(x,y,test) reduction(+:Ncirc)
for(i=0;i<num_trials; i++)
{
x = drandom(); 
y = drandom();
test = x*x + y*y;
if (test <= r*r) Ncirc++;
}
pi = 4.0 * ((double)Ncirc/(double)num_trials);
printf("\n %ld trials, pi is %lf \n",num_trials, pi);
return 0;
}
