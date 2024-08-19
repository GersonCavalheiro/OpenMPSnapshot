

#include <stdio.h>
#include <omp.h>

int main() {

int count = 0;
#pragma omp parallel num_threads(4) shared(count)
{
#pragma omp flush(count)
count++;
#pragma omp flush(count)
}

printf("Value of count: %d, construct: <flush>\n", count);
return 0;
}
