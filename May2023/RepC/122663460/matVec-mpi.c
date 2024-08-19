# include <math.h>
#ifdef USE_MPI
# include <mpi.h>
#endif
# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include "omp.h"
int main ( int argc, char *argv[] );
void timestamp ( );
int main ( int argc, char *argv[] )
{
double *a;
double *a_row;
double ans;
double *b;
int dest;
int dummy;
int i;
int ierr;
int j;
int j_one;
int k;
int m;
int master = 0;
int my_id;
int n;
int num_procs;
int num_threads;
int num_rows;
int num_workers;
double pi = 3.141592653589793;
#ifdef  USE_MPI
MPI_Status status;
#endif
int tag;
int tag_done;
double *x;
#ifdef _OPENMP
printf ("Using OpenMP library");
#endif
#ifdef USE_MPI
ierr = MPI_Init ( &argc, &argv );
if ( ierr != 0 )
{
printf ( "\n" );
printf ( "MATVEC_MPI - Fatal error!\n" );
printf ( "  MPI_Init returns nonzero IERR.\n" );
exit ( 1 );
}
ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &my_id );
ierr = MPI_Comm_size ( MPI_COMM_WORLD, &num_procs );
#else
my_id = 0;
num_procs=1;
#endif
omp_set_num_threads(16);
#ifdef USE_MPI
#pragma omp parallel
{
num_threads = omp_get_max_threads();
printf("omp_get_num_threads(): %d\n", num_threads);
}
#else
num_threads = 1;
printf("OpenMP not available. Setting num_threads: %d\n", num_threads);
#endif
if ( my_id == 0 ) 
{
timestamp ( );
printf ( "\n" );
printf ( "MATVEC - Master process:\n" );
printf ( "  C version\n" );
printf ( "  An MPI example program to compute\n" );
printf ( "  a matrix-vector product b = A * x.\n" );
printf ( "\n" );
printf ( "  Compiled on %s at %s.\n", __DATE__, __TIME__ );
printf ( "\n" );
printf ( "  The number of processes is %d.\n", num_procs );
printf ( "  The number of threads per process is %d.\n", num_threads);
}
printf ( "\n" );
printf ( "Process %d is active.\n", my_id );
m = 100;
n = 50;
tag_done = m + 1;
if ( my_id == 0 ) 
{
printf ( "\n" );
printf ( "  The number of rows is    %d.\n", m );
printf ( "  The number of columns is %d.\n", n );
}
if ( my_id == master )
{
a = ( double * ) malloc ( m * n * sizeof ( double ) );
x = ( double * ) malloc ( n * sizeof ( double ) );
b = ( double * ) malloc ( m * sizeof ( double ) );
k = 0;
#pragma omp parallel for    
for ( i = 1; i <= m; i++ ) 
{
for ( j = 1; j <= n; j++ )
{
a[k] = sqrt ( 2.0 / ( double ) ( n + 1 ) ) 
* sin ( ( double ) ( i * j ) * pi / ( double ) ( n + 1 ) );
k = k + 1;
}
}
j_one = 17;
#pragma omp parallel for num_threads(16)
for ( i = 0; i < n; i++ )
{
x[i] = sqrt ( 2.0 / ( double ) ( n + 1 ) ) 
* sin ( ( double ) ( ( i + 1 ) * j_one ) * pi / ( double ) ( n + 1 ) );
printf("thread %d doing iteration %d \n", omp_get_thread_num(), i);
}
printf ( "\n" );
printf ( "MATVEC - Master process:\n" );
printf ( "  Vector x:\n" );
printf ( "\n" );
for ( i = 0; i < n; i++ )
{
printf ( "%d %f\n", i, x[i] );
}
}
else
{
a_row = ( double * ) malloc ( n * sizeof ( double ) );
x = ( double * ) malloc ( n * sizeof ( double ) );
}
#ifdef USE_MPI
ierr = MPI_Bcast ( x, n, MPI_DOUBLE, master, MPI_COMM_WORLD );
#endif 
if ( my_id == master )
{
num_rows = 0;
for ( i = 1; i <= num_procs-1; i++ )
{
dest = i;
tag = num_rows;
k = num_rows * n;
#ifdef USE_MPI
ierr = MPI_Send ( a+k, n, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD );
#endif
num_rows = num_rows + 1;
}
num_workers = num_procs-1;
for ( ; ; )
{
#ifdef USE_MPI
ierr = MPI_Recv ( &ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE,
MPI_ANY_TAG, MPI_COMM_WORLD, &status );
#endif
#ifdef USE_MPI
tag = status.MPI_TAG;
#endif 
b[tag] = ans;
if ( num_rows < m )
{
num_rows = num_rows + 1;
#ifdef USE_MPI
dest = status.MPI_SOURCE;
#endif
tag = num_rows;
k = num_rows * n;
#ifdef USE_MPI
ierr = MPI_Send ( a+k, n, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD );
#endif
}
else
{
num_workers = num_workers - 1;
dummy = 0;
#ifdef USE_MPI
dest = status.MPI_SOURCE;
#endif
tag = tag_done;
#ifdef USE_MPI
ierr = MPI_Send ( &dummy, 1, MPI_INT, dest, tag, MPI_COMM_WORLD );
#endif
if ( num_workers == 0 )
{
break;
}
}
}
free ( a );
free ( x );
}
else
{
for ( ; ; )
{
#ifdef USE_MPI
ierr = MPI_Recv ( a_row, n, MPI_DOUBLE, master, MPI_ANY_TAG,
MPI_COMM_WORLD, &status );
tag = status.MPI_TAG;
#endif
if ( tag == tag_done ) 
{
printf ( "  Process %d shutting down.\n", my_id );
break;
}
ans = 0.0;
#pragma omp parallel for 
for ( i = 0; i < n; i++ )
{
ans = ans + a_row[i] * x[i];
}
#ifdef USE_MPI
ierr = MPI_Send ( &ans, 1, MPI_DOUBLE, master, tag, MPI_COMM_WORLD );
#endif
}
free ( a_row );
free ( x );
}
if ( my_id == master ) 
{
printf ( "\n" );
printf ( "MATVEC - Master process:\n" );
printf ( "  Product vector b = A * x\n" );
printf ( "  (Should be zero, except for a 1 in entry %d)\n", j_one-1 );
printf ( "\n" );
for ( i = 0; i < m; i++ )
{
printf ( "%d %f\n", i, b[i] );
}
free ( b );
}
#ifdef USE_MPI
ierr = MPI_Finalize ( );
#endif
if ( my_id == master ) 
{
printf ( "\n" );
printf ( "MATVEC - Master process:\n" );
printf ( "  Normal end of execution.\n" );
printf ( "\n" );
timestamp ( );
}
return 0;
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
