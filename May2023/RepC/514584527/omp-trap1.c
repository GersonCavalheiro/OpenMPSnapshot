#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
double f( double x )
{
return 4.0/(1.0 + x*x);
}
int min(int a, int b)
{
return (a < b ? a : b);
}
double trap( double a, double b, int n )
{
const int my_rank = omp_get_thread_num();
const int thread_count = omp_get_num_threads();
const double h = (b-a)/n;
const int local_n_start = (n * my_rank) / thread_count;
const int local_n_end = (n * (my_rank+1)) / thread_count;
double x = a + local_n_start*h;
double my_result = 0.0;
int i;
for ( i = local_n_start; i<local_n_end; i++ ) {
my_result += h*(f(x) + f(x+h))/2.0;
x += h;
}
return my_result;
}
int main( int argc, char* argv[] )
{
double a = 0.0, b = 1.0, result = 0.0, partial_result;
int n = 1000000;
double tstart, tstop;
if ( 4 == argc ) {
a = atof(argv[1]);
b = atof(argv[2]);
n = atol(argv[3]);
}
tstart = omp_get_wtime();
#pragma omp parallel private(partial_result)
{
partial_result = trap(a, b, n);
#pragma omp atomic
result += partial_result;
}
tstop = omp_get_wtime();
printf("Area: %f\n", result);
printf("Elapsed time %f\n", tstop - tstart);
return EXIT_SUCCESS;
}
