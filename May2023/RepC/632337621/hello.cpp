#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <omp.h>
using namespace std;
int main(int argc, char *argv[])
{
int id;
int proc_num;
int thread_num;
double wtime;
double wtime1;
double wtime2;
cout << "\n";
cout << "HELLO\n";
cout << "  C++/OpenMP version\n";
wtime1 = omp_get_wtime();
proc_num = omp_get_num_procs();
cout << "\n";
cout << "  The number of processors available:\n";
cout << "  OMP_GET_NUM_PROCS () = " << proc_num << "\n";
#pragma omp parallel private(id)
{
id = omp_get_thread_num();
thread_num = omp_get_num_threads();
if (id == 0)
{
cout << "\n";
cout << "  Calling OMP_GET_NUM_THREADS inside a\n";
cout << "  parallel region, we get the number of\n";
cout << "  threads is " << thread_num << "\n";
}
cout << "  This is process " << id << " out of " << thread_num << "\n";
}
thread_num = 2 * thread_num;
cout << "\n";
cout << "  We request " << thread_num << " threads.\n";
omp_set_num_threads(thread_num);
#pragma omp parallel private(id)
{
id = omp_get_thread_num();
thread_num = omp_get_num_threads();
if (id == 0)
{
cout << "\n";
cout << "  Calling OMP_GET_NUM_THREADS inside a\n";
cout << "  parallel region, we get the number of\n";
cout << "  threads is " << thread_num << "\n";
}
cout << "  This is process " << id << " out of " << thread_num << "\n";
}
wtime2 = omp_get_wtime();
wtime = wtime2 - wtime1;
cout << "\n";
cout << "HELLO\n";
cout << "  Normal end of execution.\n";
cout << "\n";
cout << "  Elapsed wall clock time = " << wtime << "\n";
return 0;
}
