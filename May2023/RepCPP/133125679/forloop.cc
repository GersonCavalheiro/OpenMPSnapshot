


#include <stdio.h>
#include <omp.h>

int count = 0;

void forMethod() {
#pragma omp for
for (int i = 0; i < 4; i++)
count++;
}

int main() {

#pragma omp parallel num_threads(4) shared(count)
{
forMethod();
}

printf("Value of count: %d, construct: <forloop>\n", count);
return 0;
}
