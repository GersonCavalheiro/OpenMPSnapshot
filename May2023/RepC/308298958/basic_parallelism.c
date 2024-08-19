#include <omp.h>
#include <stdio.h>
int main(){
printf("Hello from Sequential Region\n");
#pragma omp parallel
{
printf("Hello from Parallel Region\n");
}
printf("Hello from Sequential Region, again\n");
return 0;
}
