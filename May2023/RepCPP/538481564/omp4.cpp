#include <limits>
#include <omp.h>
#include <stdio.h>

using namespace std;

int main() {

int a[10] = {0, 1, 136, 3, 44498, -546, 6, -62, 8, 9};
int b[10] = {0, -1129874, 287, -445874, 4, 77476, 6, 7, 898, -4};

int min = numeric_limits<int>::max();
int max = numeric_limits<int>::min();

#pragma omp parallel num_threads(2)
{
switch (omp_get_thread_num())
{
case 0:
{
for (int i:a) 
{
if (min > i) 
min = i;
}
printf("Min element in array a = %d\n", min);
break;
}
case 1:
{
for (int i:b) 
{
if (max < i) 
max = i;
}
printf("Max element in array b = %d\n", max);
break;
}
default:
printf("Illegal thread num - %d\n", omp_get_num_threads());
}
}
}
