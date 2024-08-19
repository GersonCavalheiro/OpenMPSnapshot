#include <iostream>
#include <omp.h>
#include <math.h>

constexpr int MAX = 1e7;

using namespace std;

int main() {
printf("[Prac-code 1]\n");
#pragma omp parallel num_threads(4)
{
printf("  single  : %d Thread \n", omp_get_thread_num());
#pragma omp single
{
printf("  single   : %d Thread \n", omp_get_thread_num());
}
printf("  single  : %d Thread \n", omp_get_thread_num());
}
float* data;
int i = 0;
data = new float[MAX];

for (i = 0; i < MAX; i++) data[i] = i;

printf("\n[Prac-code 2]\n");
#pragma omp parallel
{
#pragma omp for
for (i = 0; i < 5; i++) printf("before master, tid=%d, data[%d] = %.4f\n", omp_get_thread_num(), i, data[i]);

#pragma omp master
{
for (i = 0; i < 5; i++) printf("master, tid=%d, data[%d] = %.4f\n", omp_get_thread_num(), i, data[i]);
}
#pragma omp for
for (i = 0; i < 5; i++) printf("after master, tid=%d, data[%d] = %.4f\n", omp_get_thread_num(), i, data[i]);
}
delete[] data;

printf("\n[Prac-code 3]\n");

int value[12];

for (i = 0; i < 12; i++) value[i] = i;

#pragma omp parallel for num_threads(4) ordered
for (i = 0; i < 11; i++) {
#pragma omp ordered |
{
value[i] += value[i + 1];
printf("tid: %d, %d,\n", omp_get_thread_num(), value[i]);
}
}
return 0;
}