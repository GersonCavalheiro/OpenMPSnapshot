#include <stdio.h>
#include <omp.h>
int main() {
int count = 0;
#pragma omp parallel shared(count)
{
#pragma omp single
{
count++;
}
}
printf("Value of count: %d, construct: <single>\n", count);
return 0;
}