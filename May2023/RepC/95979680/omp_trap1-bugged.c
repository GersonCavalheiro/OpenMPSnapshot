#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
void Usage(char* prog_name);
double f(double x);    
void Trap(double a, double b, int n, double* global_result_p);
double Local_Trap(double a, double b, int n);
int main(int argc, char* argv[]) {
double  global_result = 0.0;  
double  a, b;                 
int     n;                    
double elapsed = 0.0;
int     thread_count;
if (argc != 5) Usage(argv[0]);
thread_count = strtol(argv[1], NULL, 10);
a = strtol(argv[2],NULL,10);
b = strtol(argv[3],NULL,10);
n = strtol(argv[4],NULL,10);
if (n % thread_count != 0) Usage(argv[0]);
#pragma omp parallel num_threads(thread_count) reduction(max:elapsed)
{
double my_start, my_finish;
#pragma omp barrier
my_start = omp_get_wtime();
global_result += Local_Trap(a,b,n);
my_finish = omp_get_wtime();
elapsed = my_finish - my_start;
}
printf("With n = %d trapezoids, our estimate\n", n);
printf("of the integral from %f to %f = %.14e\n",
a, b, global_result);
printf("Run time: %e seconds\n", elapsed);
return 0;
}  
void Usage(char* prog_name) {
fprintf(stderr, "usage: %s <number of threads> <a> <b> <n>\n", prog_name);
fprintf(stderr, "   number of trapezoids must be evenly divisible by\n");
fprintf(stderr, "   number of threads\n");
exit(0);
}  
double f(double x) {
double return_val;
return_val = x*x;
return return_val;
}  
double Local_Trap (double a, double b, int n)
{
double  h, x, my_result;
double  local_a, local_b;
int  i, local_n;
int my_rank = omp_get_thread_num();
int thread_count = omp_get_num_threads();
h = (b-a)/n;
local_n = n/thread_count;
local_a = a + my_rank*local_n*h;
local_b = local_a + local_n*h;
my_result = (f(local_a) + f(local_b))/2.0;
for (i = 1; i <= local_n-1; i++)
{
x = local_a + i*h;
my_result += f(x);
}
my_result = my_result*h;
return (my_result);
}
void Trap(double a, double b, int n, double* global_result_p) {
double  h, x, my_result;
double  local_a, local_b;
int  i, local_n;
int my_rank = omp_get_thread_num();
int thread_count = omp_get_num_threads();
h = (b-a)/n;
local_n = n/thread_count;
local_a = a + my_rank*local_n*h;
local_b = local_a + local_n*h;
my_result = (f(local_a) + f(local_b))/2.0;
for (i = 1; i <= local_n-1; i++) {
x = local_a + i*h;
my_result += f(x);
}
my_result = my_result*h;
*global_result_p += my_result;
}  
