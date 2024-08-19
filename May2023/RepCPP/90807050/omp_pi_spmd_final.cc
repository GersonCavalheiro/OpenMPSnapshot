
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <iomanip>

#include <madthreading/types.hh>
#include <madthreading/utility/timer.hh>
#include <madthreading/threading/thread_manager.hh>
#include "../Common.hh"

using namespace mad;


int main (int, char** argv)
{
ulong_type num_steps = GetEnvNumSteps(500000000UL);
double_type step = 1.0/static_cast<double_type>(num_steps);
double_type full_sum = 0.0;

ulong_type num_threads = thread_manager::GetEnvNumThreads(1);
omp_set_num_threads(num_threads);

timer::timer t;

ulong_type i;
#pragma omp parallel private(i)
{
ulong_type id = omp_get_thread_num();
ulong_type nthreads = omp_get_num_threads();
double_type partial_sum = 0.0;

pragma_simd
for(i = id; i < num_steps; i += nthreads)
{
double_type x = (i+0.5)*step;
partial_sum += + 4.0/(1.0+x*x);
}
#pragma omp critical
full_sum += partial_sum;
}

report(num_steps, step*full_sum, t.stop_and_return(), argv[0]);

double_type pi = step * full_sum;
return (fabs(pi - M_PI) > PI_EPSILON);
}

