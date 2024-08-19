

#include <stdio.h>
#include <omp.h>

int main() {

int count = 0;
#pragma omp parallel num_threads(4) shared(count)
{
#pragma omp master
{
count++;
}
}

printf("Value of count: %d, construct: <master>\n", count);
return 0;
}
