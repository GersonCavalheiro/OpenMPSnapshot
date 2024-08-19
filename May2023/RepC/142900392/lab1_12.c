#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
int main(){
int i,n,min,max,fact;
double seq_start_time, seq_end_time,start_time, end_time,parallel_time,sequential_time;
printf("==========EXERCISE 12==========\n\t\t\tS. DHANYA ABHIRAMI\n\t\t\t16BCE0965\n");
printf("Enter n: ");
scanf("%d",&n);
int *a;
a=(int*)malloc(sizeof(int)*n);
printf("Enter array elements: ");
for(i=0;i<n;i++)
{scanf("%d",&a[i]);}
int *fib_array_seq = (int*) malloc(sizeof(int)*n);
int *fib_array = (int*) malloc(sizeof(int)*n);
printf("Sequential Computation\n");
seq_start_time = omp_get_wtime();
max = -1;
for(i=0;i<n;i++){
if(a[i]>max)
max = a[i];}			
min = 10000;
for(i=0;i<n;i++){
if(a[i]<min)
min = a[i];}			
fact = 1;
for(i=1;i<=n;i++){
fact = fact*i;			
}
int first,second;
fib_array_seq[0]=0; first=fib_array_seq[0];
fib_array_seq[1]=1; second = fib_array_seq[1];
for(i=2;i<n;i++){
fib_array_seq[i]=first+second;
first=second;
second=fib_array_seq[i];
}
seq_end_time = omp_get_wtime();
sequential_time = seq_end_time - seq_start_time;
printf("Max = %d\n",max);
printf("Min = %d\n",min);
printf("Factorial = %d\n",fact);
printf("Fibonacci Sequence: ");
for(i=0;i<n;i++)
{printf("%d ",fib_array_seq[i]);}
printf("\nExecution Time: %g\n\n",sequential_time);
printf("Parallel Computation\n");
start_time = omp_get_wtime();
#pragma omp parallel 
{
#pragma omp sections
{
#pragma omp section
{
int thread_id = omp_get_thread_num();
max = -1;
for(i=0;i<n;i++){
if(a[i]>max)
max = a[i];			
}
}
#pragma omp section
{
int thread_id = omp_get_thread_num();
printf("Thread: %d\n",thread_id);
min = 10000;
for(i=0;i<n;i++){
if(a[i]<min)
min = a[i];			
}
}
#pragma omp section
{
int thread_id = omp_get_thread_num();
printf("Thread: %d\n",thread_id);
fact = 1;
for(i=1;i<=n;i++){
fact = fact*i;			
}
}
#pragma omp section
{
int thread_id = omp_get_thread_num();
printf("Thread: %d\n",thread_id);
int first,second;
fib_array[0]=0; first=fib_array[0];
fib_array[1]=1; second = fib_array[1];
for(i=2;i<n;i++){
fib_array[i]=first+second;
first=second;
second=fib_array[i];
}
}
}
}
end_time = omp_get_wtime();
parallel_time = end_time - start_time;
printf("Max = %d\n",max);
printf("Min = %d\n",min);
printf("Factorial = %d\n",fact);
printf("Fibonacci Sequence: ");
for(i=0;i<n;i++)
{printf("%d ",fib_array[i]);}
printf("\nExecution Time: %g\n",parallel_time);
return(0);
}
