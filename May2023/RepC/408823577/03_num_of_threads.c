#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
int main( int argc, char **argv )
{
int nthreads;
#if defined(_OPENMP)
int threads_num = 1;
if ( argc > 1 )
threads_num = atoi(*(argv+1));
else
{
char *buffer = getenv("OMP_NUM_THREADS");
if ( buffer != NULL )
threads_num = atoi( buffer );
else
{
FILE *pipe = popen("cat /proc/cpuinfo | grep processor | cut -d':' -f2 | tail -n1", "r");
if ( pipe != NULL )
{
char   *string_ = NULL;
size_t  string_size;
if( getline( &string_, &string_size, pipe) > 0 )
threads_num = atoi( string_ ) + 1;
pclose(pipe);
if ( string_ != NULL )
free( string_ );
}
}
}
if ( omp_get_dynamic( ) == 0 )
{
int max_allowed = omp_get_max_threads();
if ( max_allowed != threads_num )
printf("\t>>> your OMP_DYNAMIC variable is set to FALSE\n"
"\t    your omp could not adapt the number of threads in"
"\t    a parallel region to optimize the use of the system.\n"
"\t    Still, you can adapt the number of threads by hands\n"
"\t    in each parallel region by using \"omp_set_num_threads()\"\n"
"\t    Currently, the default maximum allowed number of threads\n"
"\t    in a PR (without you modifying that) is %d\n", max_allowed );
}
omp_set_num_threads( threads_num );
#pragma omp parallel              
{   
int my_thread_id = omp_get_thread_num();
#pragma omp master
nthreads = omp_get_num_threads();
printf( "\tgreetings from thread num %d\n", my_thread_id );
}
#else
nthreads = 1;
#endif
printf(" %d thread%s greeted you from the %sparallel region\n",
nthreads, (nthreads==1)?" has":"s have", (nthreads==1)?"(non)":"" );  
return 0;
}
