#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
double clock()
{
struct timeval Tp;
int stat;
stat = gettimeofday(&Tp, NULL);
if (stat != 0)
printf("Error return from gettimeofday: %d", stat);
return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
void kernel()
{
static float x2[4000 - 1 + 2] = { 0 }, y_1[4000 - 1 + 2] = { 0 }, y_2[4000 - 1 + 2] =
{ 0 }, A[4000 - 1 + 2][4000 - 1 + 2] = { 0 }, x1[4000 - 1 + 2] = { 0 };
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i <= 4000 - 1; i++) {
for (int j = 0; j <= 4000 - 1; j++) {
#pragma omp atomic
x1[i] += A[i][j] * y_1[j];
}
}
#pragma omp for
for (int i = 0; i <= 4000 - 1; i++) {
for (int j = 0; j <= 4000 - 1; j++) {
#pragma omp atomic
x2[i] += A[j][i] * y_2[j];
}
}
}
}
int main()
{
double start = 0.0, end = 0.0;
start = clock();
kernel();
end = clock();
printf("Total time taken  = %fs\n", end - start);
return 0;
}
