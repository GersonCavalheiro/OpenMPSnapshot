

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <float.h>
#include <math.h>
#include <time.h>

#include <omp.h>
#include <assert.h>

#include "thread_level.h"

#if _WIN32||_WIN64
#include <Windows.h> 
#else
#include <unistd.h>  
#endif

void MilliSleep( unsigned milliseconds ) {
#if _WIN32||_WIN64
Sleep( milliseconds );
#else
usleep( milliseconds*1000 );
#endif 
}

const int Max_OMP_Outer_Threads = 8;

int max_outer_threads = Max_OMP_Outer_Threads;

void help_message(char *prog_name) {
fprintf(stderr, "\n%s usage:\n", prog_name);
fprintf(stderr, 
"  Parameters:\n"
"    -o<num> : max # of threads OMP should use at outer level\n"
"\n  Help:\n"
"    -h : print this help message\n");
}

void process_args(int argc, char *argv[], int *max_outer_t) {
(*max_outer_t) = omp_get_max_threads();
for (int i=1; i<argc; ++i) {  
if (argv[i][0] == '-') {
switch (argv[i][1]) {
case 'o': 
if (sscanf(&argv[i][2], "%d", max_outer_t) != 1 || *max_outer_t < 1) {
fprintf(stderr, "%s Warning: argument of -o option unacceptable: %s\n", argv[0], &argv[i][2]);
help_message(argv[0]);
}
break;
case 'h': 
help_message(argv[0]);
exit(0);
break;
default:
fprintf(stderr, "%s: Warning: command-line option ignored: %s\n", argv[0], argv[i]);
help_message(argv[0]);
break;
}
} else {
fprintf(stderr, "%s: Warning: command-line option ignored: %s\n", argv[0], argv[i]);
help_message(argv[0]);
}
}
}

int main(int argc, char *argv[]) { 
process_args(argc, argv, &max_outer_threads);
#ifdef LOG_THREADS
TotalThreadLevel.init();
#endif

double start, end;
start = omp_get_wtime( );

#pragma omp parallel num_threads(max_outer_threads)
{
int omp_thread = omp_get_thread_num();
#ifdef LOG_THREADS
if (omp_thread == 0)
TotalThreadLevel.change_level(omp_get_num_threads(), omp_outer);
#endif
if (omp_thread == 0) {
MilliSleep(3000);
#ifdef LOG_THREADS
TotalThreadLevel.change_level(-1, omp_outer);
#endif
#pragma omp parallel
{
int my_omp_thread = omp_get_thread_num();
#ifdef LOG_THREADS
if (my_omp_thread == 0)
TotalThreadLevel.change_level(omp_get_num_threads(), omp_inner);
#endif
printf("Inner thread %d nested inside outer thread %d\n", my_omp_thread, omp_thread);
#ifdef LOG_THREADS
if (my_omp_thread == 0)
TotalThreadLevel.change_level(-omp_get_num_threads(), omp_inner);
#endif
}
#ifdef LOG_THREADS
TotalThreadLevel.change_level(1, omp_outer);
#endif
}
else {
MilliSleep(6000);
}
#ifdef LOG_THREADS
if (omp_thread == 0)
TotalThreadLevel.change_level(-omp_get_num_threads(), omp_outer);
#endif
}
end = omp_get_wtime( );
printf("Simple test of nested OMP (%d outer threads max) took: %6.6f\n",
max_outer_threads, end-start);
#ifdef LOG_THREADS
TotalThreadLevel.dump();
#endif
return 0;
}
