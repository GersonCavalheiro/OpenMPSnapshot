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
static float Q[2000 - 1 + 2][2600 - 1 + 2] = { 0 }, R[2600 - 1 + 2][2600 - 1 + 2] =
{ 0 }, A[2000 - 1 + 2][2600 - 1 + 2] = { 0 }, norm = { 0 };
#pragma omp parallel
{
#pragma omp for
for (int k = 0; k <= 2600 - 1; k++) {
for (int i = 0; i <= 2000 - 1; i++) {
#pragma omp atomic
norm += A[i][k] * A[i][k];
}
#pragma omp atomic write
R[k][k] = sqrt(norm);
for (int i = 0; i <= 2000 - 1; i++) {
#pragma omp atomic write
Q[i][k] = A[i][k] / R[k][k];
}
for (int j = k + 1; j <= 2600 - 1; j++) {
for (int i = 0; i <= 2000 - 1; i++) {
#pragma omp atomic
R[k][j] += Q[i][k] * A[i][j];
}
for (int i = 0; i <= 2000 - 1; i++) {
#pragma omp atomic write
A[i][j] = A[i][j] - Q[i][k] * R[k][j];
}
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
