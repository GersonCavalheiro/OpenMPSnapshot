# include <math.h>
# include <stdio.h>
# include <stdlib.h>
int main ( );
int main ( )
{
double *b;
double d;
int i;
int it;
int m;
int n;
double r;
double t;
double *x;
double *xnew;
m = 5000;
n = 50000;
b = ( double * ) malloc ( n * sizeof ( double ) );
x = ( double * ) malloc ( n * sizeof ( double ) );
xnew = ( double * ) malloc ( n * sizeof ( double ) );
printf ( "\n" );
printf ( "JACOBI_OPENMP:\n" );
printf ( "  C/OpenMP version\n" );
printf ( "  Jacobi iteration to solve A*x=b.\n" );
printf ( "\n" );
printf ( "  Number of variables  N = %d\n", n );
printf ( "  Number of iterations M = %d\n", m );
printf ( "\n" );
printf ( "  IT     l2(dX)    l2(resid)\n" );
printf ( "\n" );
#pragma omp parallel private ( i )
{
#pragma omp for
for ( i = 0; i < n; i++ )
{
b[i] = 0.0;
}
b[n-1] = ( double ) ( n + 1 );
#pragma omp for
for ( i = 0; i < n; i++ )
{
x[i] = 0.0;
}
}
for ( it = 0; it < m; it++ )
{
#pragma omp parallel private ( i, t )
{
#pragma omp for
for ( i = 0; i < n; i++ )
{
xnew[i] = b[i];
if ( 0 < i )
{
xnew[i] = xnew[i] + x[i-1];
}
if ( i < n - 1 )
{
xnew[i] = xnew[i] + x[i+1];
}
xnew[i] = xnew[i] / 2.0;
}
d = 0.0;
#pragma omp for reduction ( + : d )
for ( i = 0; i < n; i++ )
{
d = d + pow ( x[i] - xnew[i], 2 );
}
#pragma omp for
for ( i = 0; i < n; i++ )
{
x[i] = xnew[i];
}
r = 0.0;
#pragma omp for reduction ( + : r )
for ( i = 0; i < n; i++ )
{
t = b[i] - 2.0 * x[i];
if ( 0 < i )
{
t = t + x[i-1];
}
if ( i < n - 1 )
{
t = t + x[i+1];
}
r = r + t * t;
}
#pragma omp master
{
if ( it < 10 || m - 10 < it )
{
printf ( "  %8d  %14.6g  %14.6g\n", it, sqrt ( d ), sqrt ( r ) );
}
if ( it == 9 )
{
printf ( "  Omitting intermediate results.\n" );
}
}
}
}
printf ( "\n" );
printf ( "  Part of final solution estimate:\n" );
printf ( "\n" );
for ( i = 0; i < 10; i++ )
{
printf ( "  %8d  %14.6g\n", i, x[i] );
}
printf ( "...\n" );
for ( i = n - 11; i < n; i++ )
{
printf ( "  %8d  %14.6g\n", i, x[i] );
}
free ( b );
free ( x );
free ( xnew );
printf ( "\n" );
printf ( "JACOBI_OPENMP:\n" );
printf ( "  Normal end of execution.\n" );
return 0;
}
