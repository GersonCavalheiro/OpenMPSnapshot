#include <stdio.h>
#include <omp.h>
int main() {
int count = 0;
#pragma omp parallel shared(count)
{
count++;
}
printf("Value of count: %d, construct: <parallel>\n", count);
return 0;
}