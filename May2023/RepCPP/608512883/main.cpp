#include <omp.h>

#include <iostream>

using namespace std;

int main() {
int nthreads, tid;

cout << "Program started" << endl;

omp_set_num_threads(4);

#pragma omp parallel private(nthreads, tid)
{
tid = omp_get_thread_num();
cout << "Welcome to GFG from thread = " << tid << endl;

if (tid == 0) {
nthreads = omp_get_num_threads();
cout << "Number of threads = " << nthreads << endl;
}
}

cout << "Program finished." << endl;

return 0;
}