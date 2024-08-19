#if !defined(_OPENMP)
#error you need to use OpenMP to compile this code, use the appropriated flag for your compiler
#endif
#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
#if !defined(WATCH_THREADS)
#define WATCH_THREADS 5
#endif
#if !defined(WAIT)
#define WAIT 0
#endif
#define NAME_LENGTH         2
#define LEVEL_LENGTH        2
#define TAG_LENGTH  (NAME_LENGTH + LEVEL_LENGTH + 2)
#define MIN( x, y ) ( ((x)<=(y))? (x) : (y) )
int function( char*, int );
int nesting_is_active  = 0;
int max_nesting_levels = 1;
int main( int argc, char **argv )
{
int nthreads = 4;
if ( argc > 1 )
nthreads = atoi( *(argv+1) );
if ( WAIT > 0 )
{
printf("\nwaiting %d seconds for you to start \"top -H -p %d\"\n",
WAIT, getpid());
sleep(WAIT);
}
else
printf("\nin case you want to spy me, compile with\n"
"  -DWAIT=XX\n"
"with XX large enough (for instance, ~10-15 seconds)\n\n");
#pragma omp parallel num_threads(nthreads)
#pragma omp single
{
if( nesting_is_active  = omp_get_nested() )
max_nesting_levels  = omp_get_max_active_levels();
}
if ( max_nesting_levels > 1000000 ) {
printf("somehing is strange in your max_active_level: I've got the value %u\n",
max_nesting_levels );
#if defined(__GNUC__)
printf("..in fact, you're using GCC.\n");
#endif
return 1;
}
printf("I've got that you allow %u nested levels\n", max_nesting_levels);
printf("I start recursion with %d threads\n", nthreads);
function( "00.00", nthreads );
return 0;
}
int function( char *father_name, int next )
{
#if !(defined(__ICC) || defined(__INTEL_COMPILER))
#define ENFORCE_ORDERED_OUTPUT( MSG ) {			\
int done = 0;						\
while( !done ) {						\
_Pragma("pragma omp critical(output)")			\
if( myid == order ) { printf("%s", (MSG));		\
order++; done=1;}}}
#else
#define ENFORCE_ORDERED_OUTPUT( MSG ) printf("%s", (MSG));
#endif
#define GET_LEVEL_INFO				\
int myid = omp_get_thread_num();		\
int this_level = omp_get_active_level();			\
int eff_level  = MIN( max_nesting_levels, this_level);	\
\
char buffer[max_nesting_levels+1];				\
memset( buffer, 0, max_nesting_levels+1);			\
for( int ii = 0; ii < eff_level; ii++)			\
buffer[ii] = '\t'; 
#define SETUP_MYNAME 						\
char myname[ strlen(father_name) + 1 + TAG_LENGTH + 1];	\
if ( this_level > 1 )							\
sprintf( myname, "%s-%0*d.%0*d", father_name, LEVEL_LENGTH, this_level, NAME_LENGTH, myid);	\
else									\
sprintf( myname, "%0*d.%0*d", LEVEL_LENGTH, this_level, NAME_LENGTH, myid);	\
\
char message[1000];							\
sprintf(message, "%s%s at level %d/%d\n",				\
buffer, myname,this_level, max_nesting_levels);
if ( next < 2)
{
sleep( WATCH_THREADS );
return 0;
}
int order = 0;
#pragma omp parallel num_threads(next)
{
GET_LEVEL_INFO;
SETUP_MYNAME;
ENFORCE_ORDERED_OUTPUT( message );
#pragma omp barrier
sleep( WATCH_THREADS );
function( myname, next/2 );
}
return 0;
}
