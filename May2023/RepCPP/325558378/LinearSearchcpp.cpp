#include "stdafx.h"
#include <iostream>
#include <chrono>
#include <set>
#include <stdlib.h>
#include <omp.h>

#define MAX 100

int parallel_linear_search(int* arr, int number);
int serial_linear_search(int* arr, int number);

using namespace std;

int main(){

int iter = 0, find;
int* array;

array = (int *)malloc(MAX * sizeof(int));

for (; iter < MAX; iter++) {
array[iter] = iter;
}

printf("\nEnter number to find: ");
scanf_s("%d", &find);

auto t1 = std::chrono::high_resolution_clock::now();
int index = serial_linear_search(array, find);
auto t2 = std::chrono::high_resolution_clock::now();

auto duration = t2 - t1;
if (index == -1)
printf("Not found");
else
printf("Found at %d\n", index);

std::cout << std::endl << index << " " << duration.count();

t1 = std::chrono::high_resolution_clock::now();
index = parallel_linear_search(array, find);
t2 = std::chrono::high_resolution_clock::now();

duration = t2 - t1;

std::cout << std::endl << index << " " << duration.count();

if (index == -1)
printf("Not found");
else
printf("Found at %d\n", index);

return 0;
}

int parallel_linear_search(int* arr, int number){
#pragma omp task
for (int i = 0; i < MAX; i++)
{
if (arr[i] == number)
{
return i;
}
}
return -1;
}

int serial_linear_search(int* arr, int number){
for (int i = 0; i < MAX; i++)
{
if (arr[i] == number)
{
return i;
}
}
return -1;
}