#include "iostream"
#include "omp.h"
#include "../../array_utils.h";

const int ARRAY_SIZE = 5;

void print_array(int *arr[]) {
for (int i = 0; i < ARRAY_SIZE; i++) {
print_array(arr[i], ARRAY_SIZE);
printf("\n");
}
}

int main() {

srand(time(NULL));

int **a = new int *[ARRAY_SIZE];
for (int i = 0; i < ARRAY_SIZE; i++) {
a[i] = new int[ARRAY_SIZE];
for (int j = 0; j < ARRAY_SIZE; j++) {
a[i][j] = rand() % 10;
}
}

print_array(a);

#pragma omp parallel sections num_threads(3)
{
#pragma omp section
{
int sum = 0;
for (int i = 0; i < ARRAY_SIZE; i++) {
for (int j = 0; j < ARRAY_SIZE; j++) {
sum += a[i][j];
}
}
printf("Thread %d : AVG=%d/%d=%4.2f\n",
omp_get_thread_num(), sum, (ARRAY_SIZE * ARRAY_SIZE),
(double) sum / (ARRAY_SIZE * ARRAY_SIZE));
}
#pragma omp section
{
int min = 15, max = -15;
for (int i = 0; i < ARRAY_SIZE; i++) {
for (int j = 0; j < ARRAY_SIZE; j++) {
if (a[i][j] < min) {
min = a[i][j];
}
if (a[i][j] > max) {
max = a[i][j];
}
}
}
printf("Thread %d : MIN = %d, MAX = %d\n",
omp_get_thread_num(), min, max);
}
#pragma omp section
{
int n = 0;
for (int i = 0; i < ARRAY_SIZE; i++) {
for (int j = 0; j < ARRAY_SIZE; j++) {
if (a[i][j] % 3 == 0) {
n++;
}
}
}
printf("Thread %d : number of digits, which can look like 3*K,"
" where k is Natural : %d\n",
omp_get_thread_num(), n);
}
}

delete[] a;
}
