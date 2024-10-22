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
static float std_dev[100 + 2] = { 0 }, data[100 + 2][100 + 2] = { 0 }, mean[100 + 2] = { 0 };
#pragma omp parallel
{
#pragma omp for
for (int x = 0; x <= 100; x++) {
for (int k = 0; k <= 100; k++) {
#pragma omp atomic
mean[x] += data[k][x] / 100;
}
}
#pragma omp for
for (int t = 0; t <= 100; t++) {
for (int k = 0; k <= 100; k++) {
#pragma omp atomic
std_dev[t] += ((data[k][t] - mean[t]) * (data[k][t] - mean[t])) / 99;
}
}
#pragma omp for
for (int p = 0; p <= 100; p++) {
#pragma omp atomic write
std_dev[p] = sqrt(std_dev[p]);
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
