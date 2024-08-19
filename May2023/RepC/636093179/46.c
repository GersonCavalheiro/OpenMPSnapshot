#include <stdio.h>
#include <omp.h>
#include <unistd.h>
int main()
{
int N= 10, max= 20;
#pragma omp parallel reduction(max:max)
{
#pragma omp for
for(int i= 1; i <= N; i++){
printf("%d: max= %d\n",omp_get_thread_num(),max);
if(i > max)
max= i;
}
}
printf("Calculated max= %d\n",max);
}
