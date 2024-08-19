

#include <stdio.h>
#include <omp.h>

int main() {

int count = 0;
#pragma omp parallel num_threads(4) shared(count)
{
#pragma omp sections
{
#pragma omp section
count++;

#pragma omp section
count++;
}
}

printf("Value of count: %d, construct: <sections>\n", count);
return 0;
}
