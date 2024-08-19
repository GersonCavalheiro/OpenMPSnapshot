#include <stdio.h>
#include <omp.h>
int main() {
int count = 0;
#pragma omp parallel shared(count)
{
#pragma omp critical
{
count++;
}
}
printf("Value of count: %d, construct: <critical>\n", count);
return 0;
}