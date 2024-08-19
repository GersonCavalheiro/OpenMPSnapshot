# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>
# define NX 161
# define NY 161
int main ( int argc, char *argv[] );
double r8mat_rms ( int nx, int ny, double a[NX][NY] );
void rhs ( int nx, int ny, double f[NX][NY] );
void sweep ( int nx, int ny, double dx, double dy, double f[NX][NY], 
int itold, int itnew, double u[NX][NY], double unew[NX][NY] );
void timestamp ( );
double u_exact ( double x, double y );
double uxxyy_exact ( double x, double y );
int main ( int argc, char *argv[] )
{
int converged;
double diff;
double dx;
double dy;
double error;
double f[NX][NY];
int i;
int id;
int itnew;
int itold;
int j;
int nx = NX;
int ny = NY;
double tolerance = 0.000001;
double u[NX][NY];
double u_norm;
double udiff[NX][NY];
double uexact[NX][NY];
double unew[NX][NY];
double unew_norm;
double wtime;
double x;
double y;
dx = 1.0 / ( double ) ( nx - 1 );
dy = 1.0 / ( double ) ( ny - 1 );
timestamp ( );
printf ( "\n" );
printf ( "POISSON_OPENMP:\n" );
printf ( "  C version\n" );
printf ( "  A program for solving the Poisson equation.\n" );
printf ( "\n" );
printf ( "  Use OpenMP for parallel execution.\n" );
printf ( "  The number of processors is %d\n", omp_get_num_procs ( ) );
#pragma omp parallel
{
id = omp_get_thread_num ( );
if ( id == 0 )
{
printf ( "  The maximum number of threads is %d\n", omp_get_num_threads ( ) ); 
}
}
printf ( "\n" );
printf ( "  -DEL^2 U = F(X,Y)\n" );
printf ( "\n" );
printf ( "  on the rectangle 0 <= X <= 1, 0 <= Y <= 1.\n" );
printf ( "\n" );
printf ( "  F(X,Y) = pi^2 * ( x^2 + y^2 ) * sin ( pi * x * y )\n" );
printf ( "\n" );
printf ( "  The number of interior X grid points is %d\n", nx );
printf ( "  The number of interior Y grid points is %d\n", ny );
printf ( "  The X grid spacing is %f\n", dx );
printf ( "  The Y grid spacing is %f\n", dy );
rhs ( nx, ny, f );
for ( j = 0; j < ny; j++ )
{
for ( i = 0; i < nx; i++ )
{
if ( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
{
unew[i][j] = f[i][j];
}
else
{
unew[i][j] = 0.0;
}
}
}
unew_norm = r8mat_rms ( nx, ny, unew );
for ( j = 0; j < ny; j++ )
{
y = ( double ) ( j ) / ( double ) ( ny - 1 );
for ( i = 0; i < nx; i++ )
{
x = ( double ) ( i ) / ( double ) ( nx - 1 );
uexact[i][j] = u_exact ( x, y );
}
}
u_norm = r8mat_rms ( nx, ny, uexact );
printf ( "  RMS of exact solution = %g\n", u_norm );
converged = 0;
printf ( "\n" );
printf ( "  Step    ||Unew||     ||Unew-U||     ||Unew-Exact||\n" );
printf ( "\n" );
for ( j = 0; j < ny; j++ )
{
for ( i = 0; i < nx; i++ )
{
udiff[i][j] = unew[i][j] - uexact[i][j];
}
}
error = r8mat_rms ( nx, ny, udiff );
printf ( "  %4d  %14g                  %14g\n", 0, unew_norm, error );
wtime = omp_get_wtime ( );
itnew = 0;
for ( ; ; )
{
itold = itnew;
itnew = itold + 500;
sweep ( nx, ny, dx, dy, f, itold, itnew, u, unew );
u_norm = unew_norm;
unew_norm = r8mat_rms ( nx, ny, unew );
for ( j = 0; j < ny; j++ )
{
for ( i = 0; i < nx; i++ )
{
udiff[i][j] = unew[i][j] - u[i][j];
}
}
diff = r8mat_rms ( nx, ny, udiff );
for ( j = 0; j < ny; j++ )
{
for ( i = 0; i < nx; i++ )
{
udiff[i][j] = unew[i][j] - uexact[i][j];
}
}
error = r8mat_rms ( nx, ny, udiff );
printf ( "  %4d  %14g  %14g  %14g\n", itnew, unew_norm, diff, error );
if ( diff <= tolerance )
{
converged = 1;
break;
}
}
if ( converged )
{
printf ( "  The iteration has converged.\n" );
}
else
{
printf ( "  The iteration has NOT converged.\n" );
}
wtime = omp_get_wtime ( ) - wtime;
printf ( "\n" );
printf ( "  Elapsed seconds = %g\n", wtime );
printf ( "\n" );
printf ( "POISSON_OPENMP:\n" );
printf ( "  Normal end of execution.\n" );
printf ( "\n" );
timestamp ( );
return 0;
}
double r8mat_rms ( int nx, int ny, double a[NX][NY] )
{
int i;
int j;
double v;
v = 0.0;
for ( j = 0; j < ny; j++ )
{
for ( i = 0; i < nx; i++ )
{
v = v + a[i][j] * a[i][j];
}
}
v = sqrt ( v / ( double ) ( nx * ny )  );
return v;
}
void rhs ( int nx, int ny, double f[NX][NY] )
{
double fnorm;
int i;
int j;
double x;
double y;
for ( j = 0; j < ny; j++ )
{
y = ( double ) ( j ) / ( double ) ( ny - 1 );
for ( i = 0; i < nx; i++ )
{
x = ( double ) ( i ) / ( double ) ( nx - 1 );
if ( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
{
f[i][j] = u_exact ( x, y );
}
else
{
f[i][j] = - uxxyy_exact ( x, y );
}
}
}
fnorm = r8mat_rms ( nx, ny, f );
printf ( "  RMS of F = %g\n", fnorm );
return;
}
void sweep ( int nx, int ny, double dx, double dy, double f[NX][NY], 
int itold, int itnew, double u[NX][NY], double unew[NX][NY] )
{
int i;
int it;
int j;
#pragma omp parallel shared ( dx, dy, f, itnew, itold, nx, ny, u, unew ) private ( i, it, j )
for ( it = itold + 1; it <= itnew; it++ )
{
#pragma omp for
for ( j = 0; j < ny; j++ )
{
for ( i = 0; i < nx; i++ )
{
u[i][j] = unew[i][j];
}
}
#pragma omp for
for ( j = 0; j < ny; j++ )
{
for ( i = 0; i < nx; i++ )
{
if ( i == 0 || j == 0 || i == nx - 1 || j == ny - 1 )
{
unew[i][j] = f[i][j];
}
else
{ 
unew[i][j] = 0.25 * ( 
u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy );
}
}
}
}
return;
}
void timestamp ( )
{
# define TIME_SIZE 40
static char time_buffer[TIME_SIZE];
const struct tm *tm;
time_t now;
now = time ( NULL );
tm = localtime ( &now );
strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );
printf ( "%s\n", time_buffer );
return;
# undef TIME_SIZE
}
double u_exact ( double x, double y )
{
double r8_pi = 3.141592653589793;
double value;
value = sin ( r8_pi * x * y );
return value;
}
double uxxyy_exact ( double x, double y )
{
double r8_pi = 3.141592653589793;
double value;
value = - r8_pi * r8_pi * ( x * x + y * y ) * sin ( r8_pi * x * y );
return value;
}
# undef NX
# undef NY
