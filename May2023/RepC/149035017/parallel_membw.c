#include <stdio.h>
#include <omp.h>
#define N (1<<26)
float a[N], b[N];
int main(int argc, char **argv)
{
unsigned int i = 0;
double start = 0.0;
start = omp_get_wtime();
#pragma omp parallel for
for (i=0; i<N; ++i) {
b[i] = (float)argc * a[i];
}
printf("BW: %f GBps\n", (2*N*sizeof(float))/((1<<30)*(omp_get_wtime()-start)));
return 0;
}