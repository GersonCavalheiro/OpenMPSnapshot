#include <omp.h>
#include <stdio.h>
int main(){
int my_total = 0;
#pragma omp parallel
{
for (int i=0; i<100; ++i)
my_total += omp_get_thread_num();
}
printf("The total sum = %d\n", my_total);
return 0;
}
