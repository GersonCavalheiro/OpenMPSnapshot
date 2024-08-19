#include "stdafx.h"
#include <iostream>
#include <chrono>
#include <set>

int arr[100];

int linearSearch(int number) {
for (size_t i = 0; i < 100; i++)
{
if (arr[i] == number)
{
return i;
}
}
return -1;
}

int mpSearch(int number) {
#pragma omp parallel for
for (size_t i = 0; i < 100; i++)
{
if (arr[i] == number)
{
return i;
}
}
return -1;
}

int main() {

std::set<int> numbers;

for (size_t i = 0; i < 100;)
{
arr[i++] = i;
}

auto t1 = std::chrono::high_resolution_clock::now();
int at = linearSearch(arr[99]);
auto t2 = std::chrono::high_resolution_clock::now();

auto duration = t2 - t1;

std::cout << std::endl << at << " " << duration.count();


t1 = std::chrono::high_resolution_clock::now();
at = mpSearch(arr[99]);
t2 = std::chrono::high_resolution_clock::now();

duration = t2 - t1;

std::cout << std::endl << at << " " << duration.count();
}