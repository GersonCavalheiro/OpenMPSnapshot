#include <stdio.h>
#include <omp.h>
int main ()
{
int id = -1;
#pragma omp parallel firstprivate(id)
{
id =omp_get_thread_num();
printf("(%d) Hello ",id);
printf("(%d) world!\n",id);
}
return 0;
}
