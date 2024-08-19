# include <cmath>
# include <cstdlib>
# include <ctime>
# include <fstream>
# include <iostream>
#ifdef USE_MPI
# include <mpi.h>
#endif
#ifdef _OPENMP
# include "omp.h"
#endif
using namespace std;
int main ( int argc, char *argv[] );
double boundary_condition ( double x, double time );
double initial_condition ( double x, double time );
double rhs ( double x, double time );
void timestamp ( );
void update ( int id, int p );
int main ( int argc, char *argv[] )
{
int id;
int p;
double wtime;
#ifdef USE_MPI
MPI_Init ( &argc, &argv );
MPI_Comm_rank ( MPI_COMM_WORLD, &id );
MPI_Comm_size ( MPI_COMM_WORLD, &p );
#else
id = 0;
p = 1;
#endif
if ( id == 0 )
{
timestamp ( );
cout << "\n";
cout << "HEAT_MPI:\n";
cout << "  C++/MPI version\n";
cout << "  Solve the 1D time-dependent heat equation.\n";
}
if ( id == 0 ) 
{
#ifdef USE_MPI
wtime = MPI_Wtime ( );
#else
wtime = omp_get_wtime();
#endif
}
update ( id, p );
if ( id == 0 )
{
#ifdef USE_MPI
wtime = MPI_Wtime ( ) - wtime;
#else
wtime = omp_get_wtime();
#endif
cout << "\n";       
cout << "  Wall clock elapsed seconds = " << wtime << "\n";      
}
#ifdef USE_MPI
MPI_Finalize ( );
#endif
if ( id == 0 )
{
cout << "\n";
cout << "HEAT_MPI:\n";
cout << "  Normal end of execution.\n";
cout << "\n";
timestamp ( );
}
return 0;
}
void update ( int id, int p )
{
double cfl;
double *h;
ofstream h_file;
double *h_new;
int i;
int j;
int j_min = 0;
int j_max = 400;
double k = 0.002;
int n = 100000000;
#ifdef USE_MPI
MPI_Status status;
#endif 
int tag;
double time;
double time_delta;
double time_max = 10.0;
double time_min = 0.0;
double time_new;
double *x;
double x_delta;
ofstream x_file;
double x_max = 1.0;
double x_min = 0.0;
if ( id == 0 )
{
cout << "\n";
cout << "  Compute an approximate solution to the time dependent\n";
cout << "  one dimensional heat equation:\n";
cout << "\n";
cout << "    dH/dt - K * d2H/dx2 = f(x,t)\n";
cout << "\n";
cout << "  for " << x_min << " = x_min < x < x_max = " << x_max << "\n";
cout << "\n";
cout << "  and " << time_min << " = time_min < t <= t_max = " << time_max << "\n";
cout << "\n";
cout << "  Boundary conditions are specified at x_min and x_max.\n";
cout << "  Initial conditions are specified at time_min.\n";
cout << "\n";
cout << "  The finite difference method is used to discretize the\n";
cout << "  differential equation.\n";
cout << "\n";
cout << "  This uses " << p * n << " equally spaced points in X\n";
cout << "  and " << j_max << " equally spaced points in time.\n";
cout << "\n";
cout << "  Parallel execution is done using " << p << " processors.\n";
cout << "  Domain decomposition is used.\n";
cout << "  Each processor works on " << n << " nodes, \n";
cout << "  and shares some information with its immediate neighbors.\n";
}
x = new double[n+2];
#pragma omp parallel for
for ( i = 0; i <= n + 1; i++ )
{
x[i] = ( ( double ) (         id * n + i - 1 ) * x_max
+ ( double ) ( p * n - id * n - i     ) * x_min )
/ ( double ) ( p * n              - 1 );
}
if ( p == 1 )
{
x_file.open ( "x_data.txt" );
for ( i = 1; i <= n; i++ )
{
x_file << "  " << x[i];
}
x_file << "\n";
x_file.close ( );
}
time = time_min;
h = new double[n+2];
h_new = new double[n+2];
h[0] = 0.0;
for ( i = 1; i <= n; i++ )
{
h[i] = initial_condition ( x[i], time );
}
h[n+1] = 0.0;
time_delta = ( time_max - time_min ) / ( double ) ( j_max - j_min );
x_delta = ( x_max - x_min ) / ( double ) ( p * n - 1 );
cfl = k * time_delta / x_delta / x_delta;
if ( id == 0 ) 
{
cout << "\n";
cout << "UPDATE\n";
cout << "  CFL stability criterion value = " << cfl << "\n";;
}
if ( 0.5 <= cfl ) 
{
if ( id == 0 )
{
cout << "\n";
cout << "UPDATE - Warning!\n";
cout << "  Computation cancelled!\n";
cout << "  CFL condition failed.\n";
cout << "  0.5 <= K * dT / dX / dX = " << cfl << "\n";
}
return;
}
if ( p == 1 )
{
h_file.open ( "h_data.txt" );
for ( i = 1; i <= n; i++ )
{
h_file << "  " << h[i];
}
h_file << "\n";
}
for ( j = 1; j <= j_max; j++ )
{
time_new = ( ( double ) (         j - j_min ) * time_max
+ ( double ) ( j_max - j         ) * time_min )
/ ( double ) ( j_max     - j_min );
if ( 0 < id )
{
tag = 1;
#ifdef USE_MPI
MPI_Send ( &h[1], 1, MPI_DOUBLE, id-1, tag, MPI_COMM_WORLD );
#endif
}
if ( id < p-1 )
{
tag = 1;
#ifdef USE_MPI
MPI_Recv ( &h[n+1], 1,  MPI_DOUBLE, id+1, tag, MPI_COMM_WORLD, &status );
#endif 
}
if ( id < p-1 )
{
tag = 2;
#ifdef USE_MPI
MPI_Send ( &h[n], 1, MPI_DOUBLE, id+1, tag, MPI_COMM_WORLD );
#endif
}
if ( 0 < id )
{
tag = 2;
#ifdef USE_MPI
MPI_Recv ( &h[0], 1, MPI_DOUBLE, id-1, tag, MPI_COMM_WORLD, &status );
#endif
}
#pragma omp parallel for
for ( i = 1; i <= n; i++ )
{
h_new[i] = h[i] 
+ ( time_delta * k / x_delta / x_delta ) * ( h[i-1] - 2.0 * h[i] + h[i+1] ) 
+ time_delta * rhs ( x[i], time );
}
if ( 0 == id )
{
h_new[1] = boundary_condition ( x[1], time_new );
}
if ( id == p - 1 )
{
h_new[n] = boundary_condition ( x[n], time_new );
}
time = time_new;
for ( i = 1; i <= n; i++ )
{
h[i] = h_new[i];
}
if ( p == 1 )
{
for ( i = 1; i <= n; i++ )
{
h_file << "  " << h[i];
}
h_file << "\n";
}
}
if ( p == 1 )
{
h_file.close ( );
}
delete [] h;
delete [] h_new;
delete [] x;
return;
}
double boundary_condition ( double x, double time )
{
double value;
if ( x < 0.5 )
{
value = 100.0 + 10.0 * sin ( time );
}
else
{
value = 75.0;
}
return value;
}
double initial_condition ( double x, double time )
{
double value;
value = 95.0;
return value;
}
double rhs ( double x, double time )
{
double value;
value = 0.0;
return value;
}
void timestamp ( )
{
# define TIME_SIZE 40
static char time_buffer[TIME_SIZE];
const struct std::tm *tm_ptr;
std::time_t now;
now = std::time ( NULL );
tm_ptr = std::localtime ( &now );
std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );
std::cout << time_buffer << "\n";
return;
# undef TIME_SIZE
}
