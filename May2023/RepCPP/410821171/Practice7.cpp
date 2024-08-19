#include <iostream>
#include <algorithm>
#include <omp.h>

constexpr int NUM = 100;

int main()
{
printf("1. Barrier Directives\n");

omp_set_num_threads(4);
#pragma omp parallel
{
printf("A tid = %d \n", omp_get_thread_num());
printf("B tid = %d \n", omp_get_thread_num());
}

printf("2. Nowait Directives\n");
int data[NUM][NUM] = { 0 };
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < NUM; i++) {
for (int j = 0; j < NUM; j++) {
data[i][j] = i * NUM + j + 1;
}
}

for (int i = 0; i < NUM/2; i++) {
#pragma omp for nowait
for (int j = 0; j < NUM; j++) {
std::swap(data[i][j], data[NUM - i - 1][NUM - j - 1]);
}
}
}
for (int i = 0; i < NUM; i++) {
for (int j = 0; j < NUM; j++) {
printf("data[%3d][%3d] = [%3d]\n", i, j, data[i][j]);
}
}

printf("3. Atomic Directives\n");

int i, sum = 0, local_sum;

omp_set_num_threads(12);

#pragma omp parallel private(local_sum)
{
local_sum = 0;
#pragma omp for
for (i = 1; i <= NUM; i++)
local_sum = local_sum + i;

#pragma omp atomic
sum += local_sum;
}

printf("sum = %d\n", sum);

printf("4. Critical Directives\n");

float* data1;
data1 = new float(NUM);

for (i = 0; i < NUM; i++) {
data1[i] = i;
}

float max = 0;

#pragma omp parallel for
for (i = 0; i < NUM; i++) {
#pragma omp critical(MAXVALUE)
{
if (max < data1[i]) {
max = data1[i];
}
}
}
printf("ִ밪: %.3f\n", max);
delete data1;
return 0;
}
