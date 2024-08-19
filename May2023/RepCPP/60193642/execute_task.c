


















#include "nanos.h"
#include "common.h"



void test_task_execution_overhead ( stats_t *s )
{
int i,j, nthreads = omp_get_max_threads();
double times_seq[TEST_NSAMPLES];
double times[TEST_NSAMPLES];

for ( i = 0; i < TEST_NSAMPLES; i++ ) {
times_seq[i] = GET_TIME;
for ( j = 0; j < TEST_NTASKS; j++ ) {
task(TEST_TUSECS);
}
times_seq[i] = GET_TIME - times_seq[i];
}

for ( i = 0; i < TEST_NSAMPLES; i++ ) {
times[i] = GET_TIME;
for ( j = 0; j < TEST_NTASKS; j++ ) {
#pragma omp task
task(TEST_TUSECS);
}
#pragma omp taskwait
times[i] = (((GET_TIME - times[i]) - times_seq[i]) * nthreads) / TEST_NTASKS;
}
stats( s, times, TEST_NSAMPLES);
}

int main ( int argc, char *argv[] )
{
stats_t s;

test_task_execution_overhead( &s );
print_stats ( "Execute task overhead","warm-up", &s );
test_task_execution_overhead( &s );
print_stats ( "Execute task overhead","test", &s );

return 0;
}
