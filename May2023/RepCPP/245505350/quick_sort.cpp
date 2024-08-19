#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <stdio.h>
#include <omp.h>

template<typename T>
void swp(T& a, T& b)
{ T temp = a;
a = b;
b = temp;
}

void qsrt(double* array, int left, int right) {
while (right > left) {
int iterate_left = left;
int iterate_right = right;
double pivot = array[(left + right) >> 1];

while (iterate_left <= iterate_right) {
while (array[iterate_left] < pivot) {
iterate_left += 1;
}
while (array[iterate_right] > pivot) {
iterate_right -= 1;
}
if (iterate_left <= iterate_right) {
double temp = array[iterate_left];
array[iterate_left] = array[iterate_right];
array[iterate_right] = temp;

iterate_left += 1;
iterate_right -= 1;
}
}

if ((iterate_left << 1) > left + right) {
qsrt(array, iterate_left, right);
right = iterate_left - 1;
} else {
qsrt(array, left, iterate_left - 1);
left = iterate_left;
}
}
}

void even(double* array, double* tmp, int left_part, int right_part) {
int i = 0, j;
while(i < left_part) {
tmp[i] = array[i];
i += 2;
}

double* array2 = array + left_part;
int a = 0, b = 0;
i = a;

for (i; (a < left_part) && (b < right_part); i += 2) {
array[i] = tmp[a];

if (tmp[a] <= array2[b]) {
a += 2;
} else {
array[i] = array2[b];
b += 2;
}
}

j = b;
if (a == left_part) {
while(j < right_part) {
array[i] = array2[j];
j += 2;
i += 2;
}
} else {
j = a;
while(j < left_part) {
array[i] = tmp[j];
j += 2;
i += 2;
}
}
}

void odd(double* array, double* tmp, int left_part, int right_part) {
int i = 1, j;
while(i < left_part) {
tmp[i] = array[i];
i += (1 << 1);
}

double* array2 = array + left_part;
int a = 1, b = 1;
i = a;

for (i; (a < left_part) && (b < right_part); i += (1 << 1)) {
array[i] = tmp[a];

if (tmp[a] <= array2[b]) {
a += 2;
} else {
array[i] = array2[b];
b += 2;
}
}

j = b;
if (a == left_part) {
while(j < right_part) {
array[i] = array2[j];
j += 2;
i += 2;
}
} else {
j = a;
while(j < left_part) {
array[i] = tmp[j];
j += 2;
i += 2;
}
}
}

void quick(double* array, double* tmp, int size, int part) {

#pragma omp parallel
#pragma omp single
{
if (size <= part) {
qsrt(array, 0, size - 1);
} else {
int divide = size >> 1;
int partial = divide + divide % 2;
#pragma omp task
{
quick(array, tmp, partial, part);
}

#pragma omp task
{
quick(array + partial, tmp + partial, size - partial, part);
}
#pragma omp taskwait

#pragma omp task
{
even(array, tmp, partial, size - partial);
}

#pragma omp task
{
odd(array, tmp, partial, size - partial);
}
#pragma omp taskwait

#pragma omp parallel num_threads(4)
{
#pragma omp for
for (int i = 1; i < (size + 1) >> 1; i += 1) {
if (array[i << 1] < array[(i << 1) - 1]) {
swp(array[(i << 1) - 1], array[i << 1]);
}
}
}
}
}
}

void quickSort__OMP(double* array, int threads, int size) {
double* temporary = new double[size];

int portion = size / threads;
if (size % threads)
portion += 1;

#pragma omp parallel
#pragma omp single
{
#pragma omp task
{
quick(array, temporary, size, portion);
}
}
delete[]temporary;
}

void getRandomArray(double* arr, int size) {
int i = 0;
double number;

while(i < size)
{
number = rand() / (RAND_MAX + 1.0);
arr[i] = number;
i += 1;
}
}

bool isSorted(double* ar, int size) {
const double *previous_value = ar;

while (size) {
if (*ar < *previous_value)
return false;
previous_value = ar;

++ar;
--size;
}
return true;
}


int main(void) {
srand(time(NULL));

int size = 200;
int threads = 4;

double* omp = new double[size];
double* seq = new double[size];
getRandomArray(omp, size);

for (int i = 0; i < size; i++)
{
seq[i] = omp[i];
}

double begin;
double finish;
begin = omp_get_wtime();
quickSort__OMP(omp, threads, size);
finish = omp_get_wtime();
printf("(OMP) time for quicksort = %f seconds\n", finish - begin);

clock_t start = clock();
qsrt(seq, 0, size - 1);
clock_t end = clock();
float seconds = (float)(end - start) / CLOCKS_PER_SEC;
printf("(Sequential) time for quicksort = %f \n", seconds);

if ( isSorted(omp, size) )
printf("Correctly sorted\n");
else
printf("Incorretly sorted\n");

if ( isSorted(seq, size) )
printf("Correctly sorted\n");
else
printf("Incorretly sorted\n");


for (int i = 0; i < size; i++)
{
if (omp[i] != seq[i])
{
puts("not equal"); break;
}
}

delete[]omp;
delete[]seq;

return 0;
}
