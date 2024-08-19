# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
int main ( int argc, char *argv[] );
void driver ( int m, int n, int it_max, double alpha, double omega, double tol );
void error_check ( int m, int n, double alpha, double u[], double f[] );
void jacobi ( int m, int n, double alpha, double omega, double u[], double f[], 
double tol, int it_max );
double *rhs_set ( int m, int n, double alpha );
double u_exact ( double x, double y );
double uxx_exact ( double x, double y );
double uyy_exact ( double x, double y );
int main ( int argc, char *argv[] )
{
double alpha = 0.25;
int it_max = 100;
int m = 500;
int n = 500;
double omega = 1.1;
double tol = 1.0E-08;
double wtime;
printf ( "\n" );
printf ( "HELMHOLTZ\n" );
printf ( "  C/OpenMP version\n" );
printf ( "\n" );
printf ( "  A program which solves the 2D Helmholtz equation.\n" );
printf ( "\n" );
printf ( "  This program is being run in parallel.\n" );
printf ( "\n" );
printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
printf ( "\n" );
printf ( "  The region is [-1,1] x [-1,1].\n" );
printf ( "  The number of nodes in the X direction is M = %d\n", m );
printf ( "  The number of nodes in the Y direction is N = %d\n", n );
printf ( "  Number of variables in linear system M * N  = %d\n", m * n );
printf ( "  The scalar coefficient in the Helmholtz equation is ALPHA = %f\n", 
alpha );
printf ( "  The relaxation value is OMEGA = %f\n", omega );
printf ( "  The error tolerance is TOL = %f\n", tol );
printf ( "  The maximum number of Jacobi iterations is IT_MAX = %d\n", 
it_max );
wtime = omp_get_wtime ( );
driver ( m, n, it_max, alpha, omega, tol );
wtime = omp_get_wtime ( ) - wtime;
printf ( "\n" );
printf ( "  Elapsed wall clock time = %f\n", wtime );
printf ( "\n" );
printf ( "HELMHOLTZ\n" );
printf ( "  Normal end of execution.\n" );
return 0;
}
void driver ( int m, int n, int it_max, double alpha, double omega, double tol )
{
double *f;
int i;
int j;
double *u;
f = rhs_set ( m, n, alpha );
u = ( double * ) malloc ( m * n * sizeof ( double ) );
#pragma omp parallel shared ( m, n, u ) private ( i, j )
#pragma omp for
for ( j = 0; j < n; j++ )
{
for ( i = 0; i < m; i++ )
{
u[i+j*m] = 0.0;
}
}
jacobi ( m, n, alpha, omega, u, f, tol, it_max );
error_check ( m, n, alpha, u, f );
free ( f );
free ( u );
return;
}
void error_check ( int m, int n, double alpha, double u[], double f[] )
{
double error_norm;
int i;
int j;
double u_norm;
double u_true;
double u_true_norm;
double x;
double y;
u_norm = 0.0;
#pragma omp parallel shared ( m, n, u ) private ( i, j )
#pragma omp for reduction ( + : u_norm )
for ( j = 0; j < n; j++ )
{
for ( i = 0; i < m; i++ )
{
u_norm = u_norm + u[i+j*m] * u[i+j*m];
}
}
u_norm = sqrt ( u_norm );
u_true_norm = 0.0;
error_norm = 0.0;
#pragma omp parallel shared ( m, n, u ) private ( i, j, u_true, x, y )
#pragma omp for reduction ( + : error_norm, u_true_norm)
for ( j = 0; j < n; j++ )
{
for ( i = 0; i < m; i++ )
{
x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
u_true = u_exact ( x, y );
error_norm = error_norm + ( u[i+j*m] - u_true ) * ( u[i+j*m] - u_true );
u_true_norm = u_true_norm + u_true * u_true;
}
}
error_norm = sqrt ( error_norm );
u_true_norm = sqrt ( u_true_norm );
printf ( "\n" );
printf ( "  Computed U l2 norm :       %f\n", u_norm );
printf ( "  Computed U_EXACT l2 norm : %f\n", u_true_norm );
printf ( "  Error l2 norm:             %f\n", error_norm );
return;
}
void jacobi ( int m, int n, double alpha, double omega, double u[], double f[], 
double tol, int it_max )
{
double ax;
double ay;
double b;
double dx;
double dy;
double error;
double error_norm;
int i;
int it;
int j;
double *u_old;
dx = 2.0 / ( double ) ( m - 1 );
dy = 2.0 / ( double ) ( n - 1 );
ax = - 1.0 / dx / dx;
ay = - 1.0 / dy / dy;
b  = + 2.0 / dx / dx + 2.0 / dy / dy + alpha;
u_old = ( double * ) malloc ( m * n * sizeof ( double ) );
for ( it = 1; it <= it_max; it++ )
{
error_norm = 0.0;
#pragma omp parallel shared ( m, n, u, u_old ) private ( i, j )
#pragma omp for
for ( j = 0; j < n; j++ )
{
for ( i = 0; i < m; i++ )
{
u_old[i+m*j] = u[i+m*j];
}
}
#pragma omp parallel shared ( ax, ay, b, f, m, n, omega, u, u_old ) private ( error, i, j )
#pragma omp for reduction ( + : error_norm )
for ( j = 0; j < n; j++ )
{
for ( i = 0; i < m; i++ )
{
if ( i == 0 || i == m - 1 || j == 0 || j == n - 1 )
{
error = u_old[i+j*m] - f[i+j*m];
}
else
{
error = ( ax * ( u_old[i-1+j*m] + u_old[i+1+j*m] ) 
+ ay * ( u_old[i+(j-1)*m] + u_old[i+(j+1)*m] ) 
+ b * u_old[i+j*m] - f[i+j*m] ) / b;
}
u[i+j*m] = u_old[i+j*m] - omega * error;
error_norm = error_norm + error * error;
}
}
error_norm = sqrt ( error_norm ) / ( double ) ( m * n );
printf ( "  %d  Residual RMS %e\n", it, error_norm );
if ( error_norm <= tol )
{
break;
}
}
printf ( "\n" );
printf ( "  Total number of iterations %d\n", it );
free ( u_old );
return;
}
double *rhs_set ( int m, int n, double alpha )
{
double *f;
double f_norm;
int i;
int j;
double x;
double y;
f = ( double * ) malloc ( m * n * sizeof ( double ) );
#pragma omp parallel shared ( f, m, n ) private ( i, j )
#pragma omp for
for ( j = 0; j < n; j++ )
{
for ( i = 0; i < m; i++ )
{
f[i+j*m] = 0.0;
}
}
#pragma omp parallel shared ( alpha, f, m, n ) private ( i, j, x, y )
{
#pragma omp for
for ( i = 0; i < m; i++ )
{
j = 0;
y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
f[i+j*m] = u_exact ( x, y );
}
#pragma omp for
for ( i = 0; i < m; i++ )
{
j = n - 1;
y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
f[i+j*m] = u_exact ( x, y );
}
#pragma omp for
for ( j = 0; j < n; j++ )
{
i = 0;
x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
f[i+j*m] = u_exact ( x, y );
}
#pragma omp for
for ( j = 0; j < n; j++ )
{
i = m - 1;
x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
f[i+j*m] = u_exact ( x, y );
}
#pragma omp for
for ( j = 1; j < n - 1; j++ )
{
for ( i = 1; i < m - 1; i++ )
{
x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
f[i+j*m] = - uxx_exact ( x, y ) - uyy_exact ( x, y ) + alpha * u_exact ( x, y );
}
}  
}
f_norm = 0.0;
#pragma omp parallel shared ( f, m, n ) private ( i, j )
#pragma omp for reduction ( + : f_norm )
for ( j = 0; j < n; j++ )
{
for ( i = 0; i < m; i++ )
{
f_norm = f_norm + f[i+j*m] * f[i+j*m];
}
}
f_norm = sqrt ( f_norm );
printf ( "\n" );
printf ( "  Right hand side l2 norm = %f\n", f_norm );
return f;
}
double u_exact ( double x, double y )
{
double value;
value = ( 1.0 - x * x ) * ( 1.0 - y * y );
return value;
}
double uxx_exact ( double x, double y )
{
double value;
value = -2.0 * ( 1.0 + y ) * ( 1.0 - y );
return value;
}
double uyy_exact ( double x, double y )
{
double value;
value = -2.0 * ( 1.0 + x ) * ( 1.0 - x );
return value;
}
