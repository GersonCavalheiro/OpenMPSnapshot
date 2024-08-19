#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel
{
int id = omp_get_thread_num();

printf("hello(% d) ", id);
printf("world(% d) \n", id);
}

return 0;
}