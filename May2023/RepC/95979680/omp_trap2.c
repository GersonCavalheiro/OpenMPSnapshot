#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
void Usage(char* prog_name);
double f(double x);    
double Local_trap(double a, double b, int n);
int main(int argc, char* argv[]) {
double  global_result;        
double  a, b;                 
double elapsed = 0.0;
int     n;                    
int     thread_count;
if (argc != 5) Usage(argv[0]);
thread_count = strtol(argv[1], NULL, 10);
a = strtod(argv[2],NULL);
b = strtod(argv[3],NULL);
n = strtol(argv[4],NULL,10);
if (n % thread_count != 0) Usage(argv[0]);
global_result = 0.0;
#pragma omp parallel num_threads(thread_count) reduction(max:elapsed)
{
double my_start, my_finish;
#pragma omp barrier
my_start = omp_get_wtime();
double my_result = 0.0;
my_result += Local_trap(a, b, n);
#pragma omp critical
global_result += my_result;
my_finish = omp_get_wtime();
elapsed = my_finish - my_start;
}
printf("%.10lf\n", elapsed);
return 0;
}  
void Usage(char* prog_name) {
fprintf(stderr, "Usage: %s <number of threads>\n", prog_name);
fprintf(stderr, "   number of trapezoids must be evenly divisible by\n");
fprintf(stderr, "   number of threads\n");
exit(0);
}  
double f(double x) {
double return_val;
return_val = x*x;
return return_val;
}  
double Local_trap(double a, double b, int n) {
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
return my_result;
}  
