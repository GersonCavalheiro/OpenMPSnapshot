#include <iostream>
#include <omp.h>
using namespace std;
int main()
{
int n = 5;
int a[n], b[n];
#pragma omp parallel
{
#pragma omp sections
{
#pragma omp section
{
for (int i=0; i<n; i++) 
{
a[i] = 1;
#pragma omp critical
cout << "Thread " << omp_get_thread_num() << " set a[" << i << "] = 1\n";
}
}
#pragma omp section
{
for (int i=0; i<n; i++) 
{
b[i] = 2;
#pragma omp critical
cout << "Thread " << omp_get_thread_num() << " set b[" << i << "] = 2\n";
}
}
}
}
return 0;
}
