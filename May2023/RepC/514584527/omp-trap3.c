#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
double f( double x )
{
return 4.0/(1.0 + x*x);
}
double trap( double a, double b, int n )
{
double result = 0.0;
const double h = (b-a)/n;
int i;
#pragma omp parallel for reduction(+:result)
for ( i = 0; i<n; i++ ) {
result += h*(f(a+i*h) + f(a+(i+1)*h))/2;
}
return result;
}
int main( int argc, char* argv[] )
{
double a = 0.0, b = 1.0, result;
int n = 1000000;
double tstart, tstop;
if ( 4 == argc ) {
a = atof(argv[1]);
b = atof(argv[2]);
n = atoi(argv[3]);
}
tstart = omp_get_wtime();
result = trap(a, b, n);
tstop = omp_get_wtime();
printf("Area: %f\n", result);
printf("Elapsed time %f\n", tstop - tstart);
return EXIT_SUCCESS;
}
