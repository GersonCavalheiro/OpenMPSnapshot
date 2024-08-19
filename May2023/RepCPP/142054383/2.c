#include <stdio.h>
#include <omp.h>

void printHello(int threadID)
{
printf("Hello World! %d\n", threadID);
}

int main()
{
#pragma omp parallel
{
int id = omp_get_thread_num();
printHello(id);
}

printf("There's a barrier at the end of every structured loop, all threads wait until that point");

return 0;
}