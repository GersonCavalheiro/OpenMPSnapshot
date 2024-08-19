#include <iostream>
#include <omp.h>
#include <math.h>
using namespace std;
long long a[10000000];
int main()
{
int i;
double start_t = omp_get_wtime();
#pragma omp parallel num_threads(4)
{
#pragma omp for
for (i=0; i<10000000; i++) {
a[i] = sin(i);
}
}
double end_t = omp_get_wtime();
cout << end_t - start_t << "\n";
return 0;
}
