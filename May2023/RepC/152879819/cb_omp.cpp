#include <iostream>
#include <omp.h>
using namespace std;
int main() {
#pragma omp parallel num_threads(4)
{
#pragma omp critical
{
cout << "Hello World from thread " << omp_get_thread_num() << "/" << omp_get_num_threads() <<  "\n";
}
}
return 0;
}
