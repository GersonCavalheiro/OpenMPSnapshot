#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
double calc_pi_gregory(const long max_iter_times)
{
register double cur_pi = 0;
#pragma omp parallel for reduction(+:cur_pi)
for(register int idx=1; idx<=max_iter_times; idx+=2)
{
cur_pi += (idx>>1 & 1) ? -4./idx : 4./idx;
}
return cur_pi;
}
double calc_pi_(const int max_iter_num)
{
register double x = 0;
register double pi = 0;
register double sum = 0;
register double step = 1.0 / (double) max_iter_num;
for(register int i=1; i<=max_iter_num; ++i)
{
x = (i - .5) * step;
sum += 4. / (1. + x * x);
}
pi = step * sum;
return pi;
}
double calc_pi_nilakantha(const long max_iter_times)
{
register double cur_pi = 3;
#pragma omp parallel for reduction(+:cur_pi)
for(register int idx=2; idx<=max_iter_times; idx+=2)
{
cur_pi += (idx>>1 & 1) ? 4./(idx*(idx+1)*(idx+2)) : -4./(idx*(idx+1)*(idx+2));
}
return cur_pi;
}
#define MIN_BLOCK  (1024*1024*256)
double pi_comp(register int start_step_num, register int finish_step_num, register double step)
{  
register int i, block_idx;
register double x, sum = 0.0, sum1, sum2;
if(finish_step_num-start_step_num < MIN_BLOCK)
{   
for(i = start_step_num; i < finish_step_num; ++i)
{   
x = (i+0.5) * step;
sum = sum + 4.0 / (1.0 + x * x); 
}   
}   
else
{   
block_idx = finish_step_num - start_step_num;
sum1 = pi_comp(start_step_num,              finish_step_num-block_idx/2, step);
sum2 = pi_comp(finish_step_num-block_idx/2, finish_step_num,             step);
sum = sum1 + sum2;
}
return sum;
} 
int main()
{
long max_iter_times = 1000000000;
double start_time = 0.0;
double run_time = 0.0;
double pi = 0.0;
printf("---- method1: %s ----\n", "gregory");
start_time = omp_get_wtime();
pi = calc_pi_gregory(max_iter_times);
run_time = omp_get_wtime() - start_time;
printf("pi: %.80f\n", pi);
printf("time: %lf second(s)\n", run_time);
printf("steps: %ld\n\n", max_iter_times);
printf("---- method2: %s ----\n", "gregory-likely");
start_time = omp_get_wtime();
pi = calc_pi_(max_iter_times);
run_time = omp_get_wtime() - start_time;
printf("pi: %.80f\n", pi);
printf("time: %lf second(s)\n", run_time);
printf("steps: %ld\n\n", max_iter_times);
printf("---- method3: %s ----\n", "nilakantha");
start_time = omp_get_wtime();
pi = calc_pi_nilakantha(max_iter_times);
run_time = omp_get_wtime() - start_time;
printf("pi: %.80f\n", pi);
printf("time: %lf second(s)\n", run_time);
printf("steps: %ld\n\n", max_iter_times);
printf("---- method4: %s ----\n", "divide and conquer algorithm and recurrsion");
max_iter_times = 1024*1024*1024;
double step = 1.0/(double) max_iter_times;
double sum = 0.0;
start_time = omp_get_wtime();
sum = pi_comp(0, max_iter_times, step);
pi = step * sum;
run_time = omp_get_wtime() - start_time;
printf("pi: %.80f\n", pi);
printf("time: %lf second(s)\n", run_time);
printf("steps: %ld\n\n", max_iter_times);
return 0;
}
