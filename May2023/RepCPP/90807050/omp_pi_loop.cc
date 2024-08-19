
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <iomanip>
#include <thread>

#include <madthreading/types.hh>
#include <madthreading/utility/timer.hh>
#include <madthreading/threading/thread_manager.hh>
#include "../Common.hh"

using namespace mad;


int main (int, char** argv)
{
ulong_type num_steps = GetEnvNumSteps(500000000UL);
double_type step = 1.0/static_cast<double_type>(num_steps);
ulong_type num_threads = thread_manager::GetEnvNumThreads(1);
omp_set_num_threads(num_threads);
double_type sum = 0.0;

timer::timer t;

#pragma omp parallel
{
#pragma omp for reduction(+:sum)
for(ulong_type i = 0; i < num_steps; ++i)
{
double_type x = (i-0.5)*step;
sum = sum + 4.0/(1.0+x*x);
}
}

report(num_steps, step*sum, t.stop_and_return(), argv[0]);

double_type pi = step * sum;
return (fabs(pi - M_PI) > PI_EPSILON);
}

