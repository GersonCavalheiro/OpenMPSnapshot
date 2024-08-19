
#include <iostream>
#include <omp.h>

int main() {
int a[10] = {0, 5, -3, 6, -42, 12, -30, 4, 0, 1};
int b[10] = {23, 42, -2, -54, -123, 0, -1, -1, 1, 2};

int min = INT_MAX;
int max = INT_MIN;

#pragma omp parallel num_threads(2)
{
#pragma omp master
{
for (int e: a) {
if (e < min) min = e;
}
std::cout << "Min value in a array = " + std::to_string(min) + ". \n";
}

if (omp_get_thread_num() == 1) {
for (int e: b) {
if (e > max) max = e;
}
std::cout << "Max value in b array = " + std::to_string(max) + ". \n";
}
}
return 0;
}
