# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
int main ( int argc, char *argv[] );
void r8_mxm ( int l, int m, int n );
double r8_uniform_01 ( int *seed );
int main ( int argc, char *argv[] )
{
int l;
int m;
int n;
printf ( "\n" );
printf ( "MXM\n" );
printf ( "  C/OpenMP version.\n" );
printf ( "\n" );
printf ( "  Matrix multiplication tests.\n" );
printf ( "\n" );
printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
l = 1000;
m = 1000;
n = 1000;
r8_mxm ( l, m, n );
printf ( "\n" );
printf ( "MXM:\n" );
printf ( "  Normal end of execution.\n" );
return 0;
}
void r8_mxm ( int l, int m, int n )
{
double *a;
double *b;
double *c;
int i;
int j;
int k;
int ops;
double rate;
int seed;
double time_begin;
double time_elapsed;
double time_stop;
a = ( double * ) malloc ( l * n * sizeof ( double ) );
b = ( double * ) malloc ( l * m * sizeof ( double ) );
c = ( double * ) malloc ( m * n * sizeof ( double ) );
seed = 123456789;
for ( k = 0; k < l * m; k++ )
{
b[k] = r8_uniform_01 ( &seed );
}
for ( k = 0; k < m * n; k++ )
{
c[k] = r8_uniform_01 ( &seed );
}
time_begin = omp_get_wtime ( );
#pragma omp parallel shared ( a, b, c, l, m, n ) private ( i, j, k )
#pragma omp for
for ( j = 0; j < n; j++)
{
for ( i = 0; i < l; i++ )
{
a[i+j*l] = 0.0;
for ( k = 0; k < m; k++ )
{
a[i+j*l] = a[i+j*l] + b[i+k*l] * c[k+j*m];
}
}
}
time_stop = omp_get_wtime ( );
ops = l * n * ( 2 * m );
time_elapsed = time_stop - time_begin;
rate = ( double ) ( ops ) / time_elapsed / 1000000.0;
printf ( "\n" );
printf ( "R8_MXM matrix multiplication timing.\n" );
printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
printf ( "  L = %d\n", l );
printf ( "  M = %d\n", m );
printf ( "  N = %d\n", n );
printf ( "  Floating point OPS roughly %d\n", ops );
printf ( "  Elapsed time dT = %f\n", time_elapsed );
printf ( "  Rate = MegaOPS/dT = %f\n", rate );
free ( a );
free ( b );
free ( c );
return;
}
double r8_uniform_01 ( int *seed )
{
int k;
double r;
k = *seed / 127773;
*seed = 16807 * ( *seed - k * 127773 ) - k * 2836;
if ( *seed < 0 )
{
*seed = *seed + 2147483647;
}
r = ( double ) ( *seed ) * 4.656612875E-10;
return r;
}
