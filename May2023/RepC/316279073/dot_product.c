# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
int main ( int argc, char *argv[] );
double test01 ( int n, double x[], double y[] );
double test02 ( int n, double x[], double y[] );
int main ( int argc, char *argv[] )
{
double factor;
int i,j;
int n;
int repeat = 100;
double avg_time_p = 0;
double avg_time_s = 0;
double wtime;
double *x;
double xdoty;
double *y;
printf ( "\n" );
printf ( "DOT_PRODUCT\n" );
printf ( "  C/OpenMP version\n" );
printf ( "\n" );
printf ( "  A program which computes a vector dot product.\n" );
printf ( "\n" );
printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
n = 100;
while ( n < 10000000 )
{
n = n * 10;
x = ( double * ) malloc ( n * sizeof ( double ) );
y = ( double * ) malloc ( n * sizeof ( double ) );
factor = ( double ) ( n );
factor = 1.0 / sqrt ( 2.0 * factor * factor + 3 * factor + 1.0 );
for ( i = 0; i < n; i++ )
{
x[i] = ( i + 1 ) * factor;
}
for ( i = 0; i < n; i++ )
{
y[i] = ( i + 1 ) * 6 * factor;
}
printf ( "\n" );
avg_time_s = 0;
for(j=0;j<repeat;j++){
wtime = omp_get_wtime ( );
xdoty = test01 ( n, x, y );
wtime = omp_get_wtime ( ) - wtime;
avg_time_s += wtime;
}
printf ( "  Sequential avg %15.10f\n", avg_time_s/((double) repeat));
avg_time_p = 0;
for(j=0;j<repeat;j++){
wtime = omp_get_wtime ( );
xdoty = test02 ( n, x, y );
wtime = omp_get_wtime ( ) - wtime;
avg_time_p += wtime;
}
printf ( "  Parallel   avg %15.10f\n", avg_time_p/((double) repeat));
printf ( "  Speedup (\%)    %15.10f\n", (avg_time_s/avg_time_p*100.0)-100.0);
free ( x );
free ( y );
}
printf ( "\n" );
printf ( "DOT_PRODUCT\n" );
printf ( "  Normal end of execution.\n" );
return 0;
}
double test01 ( int n, double x[], double y[] )
{
int i;
double xdoty;
xdoty = 0.0;
for ( i = 0; i < n; i++ )
{
xdoty = xdoty + x[i] * y[i];
}
return xdoty;
}
double test02 ( int n, double x[], double y[] )
{
int i;
double xdoty;
xdoty = 0.0;
#pragma omp parallel shared ( n, x, y ) private ( i )
#pragma omp for reduction ( + : xdoty )
for ( i = 0; i < n; i++ )
{
xdoty = xdoty + x[i] * y[i];
}
return xdoty;
}
