#include <stdio.h>
#include <omp.h>
void printHello(int threadID)
{
printf("Hello World! %d\n",threadID );
}
int main()
{       omp_set_num_threads(4);
#pragma omp parallel
{
int ID=omp_get_thread_num();
printHello(ID);
}
return 0;
}
