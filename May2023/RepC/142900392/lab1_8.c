#include <omp.h>
#include<stdio.h>
int main()
{
printf("==========EXERCISES 8==========\n");
printf("	Parallelising For Loop		");
printf("\n\t\t\t\tS.DHANYA ABHIRAMI \n\t\t\t\t16BCE0965 \n\n");
int i,thread_num,iter;
printf("Printing numbers from 1 to n\n");
printf("Enter n: ");
scanf("%d",&iter);
#pragma omp parallel 
{
#pragma omp for
for(i=1;i<=iter;i++){
printf("\n%d\t",i);
thread_num=omp_get_thread_num();
printf("Thread ID: %d\n",thread_num);
}	
}
return 0;
}
