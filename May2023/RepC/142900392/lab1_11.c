# include <stdio.h>
# include <omp.h>
# include<stdlib.h>
int main(){
int i,first_fifty_sum = 0,next_fifty_sum = 0;
printf("==========SUM OF SQUARES FIRST 100 NATURAL NUMBERS==========\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");
double start_time=omp_get_wtime();
#pragma omp parallel sections
{
#pragma omp section
{
int thread_id = omp_get_thread_num();
printf("Thread ID: %d\n",thread_id );
for(i=1;i<=50;i++)
{
first_fifty_sum+=i*i;
}
printf("Partial sum = %d\n", first_fifty_sum);
}
#pragma omp section
{
int thread_id = omp_get_thread_num();
printf("Thread ID: %d\n",thread_id );
for(i=51;i<=100;i++)
{
next_fifty_sum+=i*i;
}
printf("Partial sum = %d\n", next_fifty_sum);
}
}
printf("\nTime: %lf\n", omp_get_wtime()-start_time);
printf("\nRequired sum: %d\n",first_fifty_sum+next_fifty_sum);
return(0);
}