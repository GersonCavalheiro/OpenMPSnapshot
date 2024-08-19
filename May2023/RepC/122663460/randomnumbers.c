# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include <time.h>
int main ( );
void monte_carlo ( int n, int *seed );
double random_value ( int *seed );
void timestamp ( );
int main ( void )
{
int n;
int seed;
timestamp ( );
printf ( "\n" );
printf ( "RANDOM_OPENMP\n" );
printf ( "  C version\n" );
printf ( "  An OpenMP program using random numbers.\n" );
printf ( "  The random numbers depend on a seed.\n" );
printf ( "  We need to insure that each OpenMP thread\n" );
printf ( "  starts with a different seed.\n" );
printf ( "\n" );
printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
n = 100;
seed = 123456789;
monte_carlo ( n, &seed );
printf ( "\n" );
printf ( "RANDOM_OPENMP\n" );
printf ( "  Normal end of execution.\n" );
printf ( "\n" );
timestamp ( );
return 0;
}
void monte_carlo ( int n, int *seed )
{
int i;
int my_id;
int my_seed;
double *x;
x = ( double * ) malloc ( n * sizeof ( double ) );
#pragma omp master
{
printf ( "\n" );
printf ( "  Thread   Seed  I   X(I)\n" );
printf ( "\n" );
}
#pragma omp parallel private ( i, my_id, my_seed ) shared ( n, x )
{
my_id = omp_get_thread_num ( );
my_seed = *seed + my_id;
printf ( "  %6d  %12d\n", my_id, my_seed );
#pragma omp for
for ( i = 0; i < n; i++ )
{
x[i] = random_value ( &my_seed );
printf ( "  %6d  %12d  %6d  %14.6g\n", my_id, my_seed, i, x[i] );
}
}
free ( x );
return;
}
double random_value ( int *seed )
{
double r;
*seed = ( *seed % 65536 );
*seed = ( ( 3125 * *seed ) % 65536 );
r = ( double ) ( *seed ) / 65536.0;
return r;
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
