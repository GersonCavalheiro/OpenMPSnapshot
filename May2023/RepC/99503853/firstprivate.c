#include <stdio.h>
#include <omp.h>
int main()
{
int i,value=100;
#pragma omp parallel firstprivate(value)
{
#pragma omp for
for(i=0;i<4;i++)
{
value+=1;
printf("value=%d increased by thread=%d\n",value,omp_get_thread_num());
}
}
printf("outside parallel region value=%d\n",value);
}
