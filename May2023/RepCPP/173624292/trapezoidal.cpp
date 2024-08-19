
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <math.h>

double value_at(double x)
{
return sin(x);
}

double trapezoidal(double a, double b, int n,int nthreads)
{
double h = (b - a) / n;

double s = value_at(a) + value_at(b);

int i;
int myid ;
#pragma omp parallel for schedule(dynamic,nthreads)  default(none) private(i) shared(n,a,h,nthreads)  reduction(+:s)
for (i = 0; i < n; i++) {

s =s + 2 * value_at(a + i * h);
}

return (h / 2)*s;
}

int main()
{   

double x0 = 0;
double xn = 3.14159;

int n = 1500;
int nthreads= 6;
omp_set_num_threads(nthreads);
printf("Value of integral is %f\n",
(trapezoidal(x0, xn, n,nthreads)));
return 0;
}
