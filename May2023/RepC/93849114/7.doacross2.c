#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>	
#define M 4
int main() 
{
float a1[M][M], b1[M][M];
omp_set_num_threads(4);
#pragma omp parallel 
{
#pragma omp for ordered(1) schedule(dynamic)
for (int i=0; i<M; i++) {
for (int j=0; j<M; j++) {
#pragma omp ordered depend(sink: i-1)
{
printf("Computing loop 1 iteration %d %d\n", i, j);
a1[i][j] = 3.45;
}
#pragma omp ordered depend(source)
}
for (int j=2; j<M; j++) {
#pragma omp ordered depend(sink: i)
{
printf("Computing loop 2 iteration %d %d\n", i, j);
b1[i][j] = a1[i][j-1] * 0.978;
}
#pragma omp ordered depend(source)
}
}
}
return 0;
}
