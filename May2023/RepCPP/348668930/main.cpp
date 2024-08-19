#include <iostream>
#include <omp.h>
#include <x86intrin.h>
#include "ipp.h"

#define ARRAY_SIZE 1048576 
#define COEFFICIENT_PARALLEL_SORT 0.13f
#define NUM_OF_THREADS 8

using namespace std;

void printGroupMemberNames() {
cout << "Group Members:" << endl;
cout << "1 - Aryan Haddadi 810196448" << endl;
cout << "2 - Iman Moradi 810196560" << endl;
cout << "-------------" << endl;
}


Ipp64u Q1Serial(float* arr) {
Ipp64u start, end;

float max = -1;
int maxIndex;

start = ippGetCpuClocks();

for(int i = 0; i < ARRAY_SIZE; i++) {
if (arr[i] > max) {
max = arr[i];
maxIndex = i;
}
}

end = ippGetCpuClocks();


cout << "Maximum Element in Serial Calculation is " << max << endl;
cout << "Maximum Element Index in Serial Calculation is " << maxIndex << endl;
return end - start;

}

Ipp64u Q1Parallel(float* arr) {
Ipp64u start, end;

float maxTotal = -1;
int maxIndexTotal;

start = ippGetCpuClocks();

omp_lock_t lock;
omp_init_lock(&lock);

int i;
float maxThread;
int maxIndexThread;
#pragma omp parallel shared(arr, start, end, maxTotal, maxIndexTotal, lock) private(i, maxThread, maxIndexThread) num_threads(NUM_OF_THREADS)
{   

maxThread = -1;
#pragma omp for nowait 
for(i = 0; i < ARRAY_SIZE; i++) {
if (arr[i] > maxThread) {
maxThread = arr[i];
maxIndexThread = i;
}
}

omp_set_lock(&lock);
if (maxThread > maxTotal) {
maxTotal = maxThread;
maxIndexTotal = maxIndexThread;
}
omp_unset_lock(&lock);

}
omp_destroy_lock(&lock);

end = ippGetCpuClocks();


cout << "Maximum Element in Parallel Calculation is " << maxTotal << endl;
cout << "Maximum Element Index in Parallel Calculation is " << maxIndexTotal << endl;
return end - start;
}



void Q1() {
cout << "Question 1" << endl << endl;

Ipp64u serialCalculationTime, parallelCalculationTime;

srand((unsigned int)time(NULL));
float *arr = new float[ARRAY_SIZE];
for(int i = 0; i < ARRAY_SIZE; i++) {
arr[i] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX) * ARRAY_SIZE;
}

serialCalculationTime = Q1Serial(arr);
cout << "Serial Calculation Time is " << serialCalculationTime << " Clock Cycles. " << endl;

parallelCalculationTime = Q1Parallel(arr);
cout << "Parallel Calculation Time is " << parallelCalculationTime << " Clock Cycles. " << endl;

cout << "Speedup is " << float(serialCalculationTime) / float(parallelCalculationTime) << endl;
cout << "-------------" << endl;
}


void arrcpy(float* clone, float* origin, int size) {
for (int i = 0; i < size; i++)
clone[i] = origin[i];
}

void swap(float* numbers, int firstIndex, int secondIndex) {
float temp = numbers[firstIndex];
numbers[firstIndex] = numbers[secondIndex];
numbers[secondIndex] = temp;	
}

int partition(float* numbers, int startIndex, int endIndex) {
int pivotIndex = startIndex;
float pivotNumber = numbers[pivotIndex];
for (int i = startIndex + 1; i <= endIndex; i++) {
if (numbers[i] <= pivotNumber) {
pivotIndex++;
swap(numbers, i, pivotIndex);
}
}
swap(numbers, pivotIndex, startIndex);
return pivotIndex;
}

void serialQuickSort(float* numbers, int startIndex, int endIndex) {
if (startIndex >= endIndex)
return;
int pivot = partition(numbers, startIndex, endIndex);
serialQuickSort(numbers, startIndex, pivot - 1);
serialQuickSort(numbers, pivot + 1, endIndex);
}


void parallelQuickSort(float* numbers, int startIndex, int endIndex, int size) {
if (startIndex >= endIndex)
return;

int pivot =	partition(numbers, startIndex, endIndex);

#pragma omp task
{
parallelQuickSort(numbers, startIndex, pivot - 1, size);
}
#pragma omp task
{
parallelQuickSort(numbers, pivot + 1, endIndex, size);
}
}

Ipp64u Q2Serial(float* arr) {
Ipp64u start, end;

start = ippGetCpuClocks();

serialQuickSort(arr, 0, ARRAY_SIZE - 1);

end = ippGetCpuClocks();

return end - start;
}

Ipp64u Q2Parallel(float* arr) {
Ipp64u start, end;

start = ippGetCpuClocks();

#pragma omp parallel num_threads(NUM_OF_THREADS)
{
#pragma omp single
{
parallelQuickSort(arr, 0, ARRAY_SIZE - 1, ARRAY_SIZE);
}
}

end = ippGetCpuClocks();

return end - start;
}

void Q2() {
cout << "Question 2" << endl << endl;;

Ipp64u serialCalculationTime, parallelCalculationTime;

srand((unsigned int)time(NULL));
float *arr = new float[ARRAY_SIZE];
for(int i = 0; i < ARRAY_SIZE; i++) {
arr[i] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX) * ARRAY_SIZE;
}

float* serialArr = new float[ARRAY_SIZE];
arrcpy(serialArr, arr, ARRAY_SIZE);
serialCalculationTime = Q2Serial(serialArr);

float* parallelArr = new float[ARRAY_SIZE];
arrcpy(parallelArr, arr, ARRAY_SIZE);
parallelCalculationTime = Q2Parallel(parallelArr);

for (int i = 0; i < ARRAY_SIZE - 1; i++) {
if (serialArr[i] > serialArr[i+1]) {
printf("\u001b[31mERROR :: Serial Sort Checking Error\u001b[0m: Array[%d] = %f > %f = Array[%d]\n", i, serialArr[i], serialArr[i+1], i+1);
return;
}
if (parallelArr[i] > parallelArr[i+1]) {
printf("\u001b[31mERROR :: Parallel Sort Checking Error\u001b[0m: Array[%d] = %f > %f = Array[%d]\n", i, parallelArr[i], parallelArr[i+1], i+1);
return;
}
if (parallelArr[i] != serialArr[i]) {
printf("\u001b[31mERROR :: Parallel Serial Equality Checking Error\u001b[0m: Array[%d] = %f > %f = Array[%d]\n", i, parallelArr[i], serialArr[i], i+1);
return;
}
}

cout << "Sorting Results Verification was \u001b[32mSuccessful\u001b[0m." << endl;
cout << "Serial Calculation Time is " << serialCalculationTime << " Clock Cycles. " << endl;
cout << "Parallel Calculation Time is " << parallelCalculationTime << " Clock Cycles. " << endl;
cout << "Speedup is " << float(serialCalculationTime) / float(parallelCalculationTime) << endl;

}

int main() {
printGroupMemberNames();
Q1();
Q2();

return 0;
}