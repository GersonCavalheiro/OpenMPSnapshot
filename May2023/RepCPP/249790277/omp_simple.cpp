

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <float.h>
#include <math.h>
#include <time.h>

#include <omp.h>
#include <assert.h>

#include "thread_level.h"

#include "tbb/task.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

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

using namespace std;
using namespace tbb;

const int Max_TBB_Threads = 16;
const int Max_OMP_Threads = 16;

int max_tbb_threads = Max_TBB_Threads;
int max_omp_threads = Max_OMP_Threads;

void help_message(char *prog_name) {
fprintf(stderr, "\n%s usage:\n", prog_name);
fprintf(stderr, 
"  Parameters:\n"
"    -t<num> : max # of threads TBB should use\n"
"    -o<num> : max # of threads OMP should use\n"
"\n  Help:\n"
"    -h : print this help message\n");
}

void process_args(int argc, char *argv[], int *max_tbb_t, int *max_omp_t) {
for (int i=1; i<argc; ++i) {  
if (argv[i][0] == '-') {
switch (argv[i][1]) {
case 't': 
if (sscanf(&argv[i][2], "%d", max_tbb_t) != 1 || *max_tbb_t < 1) {
fprintf(stderr, "%s Warning: argument of -t option unacceptable: %s\n", argv[0], &argv[i][2]);
help_message(argv[0]);
}
break;
case 'o': 
if (sscanf(&argv[i][2], "%d", max_omp_t) != 1 || *max_omp_t < 1) {
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
process_args(argc, argv, &max_tbb_threads, &max_omp_threads);
TotalThreadLevel.init();

double start, end;
start = omp_get_wtime();

#pragma omp parallel num_threads(max_omp_threads)
{
int omp_thread = omp_get_thread_num();
#ifdef LOG_THREADS
if (omp_thread == 0)
TotalThreadLevel.change_level(omp_get_num_threads(), omp_outer);
#endif
task_scheduler_init phase(max_tbb_threads);
if (omp_thread == 0) {
MilliSleep(3000);
#ifdef LOG_THREADS
TotalThreadLevel.change_level(-1, omp_outer);
#endif
parallel_for(blocked_range<size_t>(0, 1000), 
[=](const blocked_range<size_t>& range) {
#ifdef LOG_THREADS
TotalThreadLevel.change_level(1, tbb_inner);
#endif
#pragma ivdep
for (size_t i=range.begin(); i!=range.end(); ++i) {
if (i==range.begin())
printf("TBB range starting at %d on OMP thread %d\n", (int)i, omp_thread);
}
#ifdef LOG_THREADS
TotalThreadLevel.change_level(-1, tbb_inner);
#endif
}, auto_partitioner());
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
end = omp_get_wtime();
printf("Simple test of OMP (%d threads max) with TBB (%d threads max) inside took: %6.6f\n",
max_omp_threads, max_tbb_threads, end-start);
#ifdef LOG_THREADS
TotalThreadLevel.dump();
#endif
return 0;
}
