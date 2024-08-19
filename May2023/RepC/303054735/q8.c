#include <omp.h>
#include <stdio.h>
#define SIZE 100000
int flag = 0;
void fill_rand(int N,double A[])
{
for(int i=0;i<N;++i)
A[i] = 1;
printf("Producer populated data\n");
#pragma omp flush
flag = 1;
#pragma omp flush(flag)
}
double Sum_array(int N,double A[])
{
double sum = 0.0;
int p_flag;
while(1)
{
p_flag = 0;
#pragma omp flush(flag)
p_flag = flag;
if(p_flag)
break;
}
#pragma omp flush
for(int i=0;i<N;++i)
sum = sum + A[i];
printf("Consumer calculated Array sum\n" );
return sum;
}
double seq_prod_cons()
{
double A[SIZE];
fill_rand(SIZE,A);
double sum = Sum_array(SIZE,A);
return sum;
}
double parallel_prod_cons()
{
double A[SIZE];
double  sum = 0.0;
omp_set_num_threads(2);
#pragma omp parallel sections
{
#pragma omp section
fill_rand(SIZE,A);
#pragma omp section
sum = Sum_array(SIZE,A);
}
return sum;
}
int main()
{
double time_taken_seq,time_taken_parallel,sum=0.0;
time_taken_seq = omp_get_wtime();
sum = seq_prod_cons();
time_taken_seq = omp_get_wtime() - time_taken_seq;
printf("In %lf seconds, Sequential code gives sum : %lf \n",time_taken_seq,sum);
time_taken_parallel = omp_get_wtime();
sum = parallel_prod_cons();
time_taken_parallel = omp_get_wtime() - time_taken_parallel;
printf("In %lf seconds, Parallel code gives sum : %lf \n",time_taken_parallel,sum);
printf("Speed up : %lf\n", time_taken_parallel/time_taken_seq);
}