#include <stdio.h>
#include <omp.h>
int main() {
int count = 0;
#pragma omp parallel shared(count)
{
#pragma omp atomic
count++;
}
printf("Value of count: %d, construct: <atomic>\n", count);
return 0;
}
