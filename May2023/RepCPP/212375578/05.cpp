#include <cstdio>
#include <omp.h>

#include "lib.h"

int main() {
const int n = 6, m = 6;
int** d = random2dArray(n, m);
print2dArray(d, n, m);

int min = d[0][0], max = d[0][0];
double sum = 0;
double avg;
int multiples_of_3 = 0;

#pragma omp parallel sections
{
#pragma omp section
{
for (int i = 0; i < n; i++) {
for (int j = 0; j < m; j++) {
sum += d[i][j];
}
}
avg = 1.0 * sum / (6 * 8);
printf("avg = %f by thread#%d\n", avg, omp_get_thread_num());
}

#pragma omp section
{
for (int i = 0; i < n; i++) {
for (int j = 0; j < m; j++) {
if (d[i][j] < min) {
min = d[i][j];
}
if (d[i][j] > max) {
max = d[i][j];
}
}
}
printf("min = %d, max = %d by thread#%d\n", min, max, omp_get_thread_num());
}

#pragma omp section
{
for (int i = 0; i < n; i++) {
for (int j = 0; j < m; j++) {
if (d[i][j] % 3 == 0) {
++multiples_of_3;
}
}
}
printf("multiples of 3 count = %d by thread#%d\n", multiples_of_3, omp_get_thread_num());
}
}
}
