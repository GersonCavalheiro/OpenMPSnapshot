#include <stdio.h>
#include <omp.h>
int main ()
{
int id;
#pragma omp parallel
{
#pragma omp critical
{
id =omp_get_thread_num();
printf("(%d) Hello ",id);
printf("(%d) world!\n",id);
} 
}
return 0;
}
