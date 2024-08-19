#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
double f(double x);
double trap_serial(double a, double b, int n);
double trap(double a, double b, int n);
int main(int argc, char* argv[]) {
double  global_result = 0.0;  
double  a, b;                 
int     n;                    
int     thread_count;
thread_count = 4;
a = 1;
b = 5;
n = 1e8; 
if (argc > 1) {
thread_count = atoi(argv[1]);
}
if (argc > 4) {
thread_count = atoi(argv[1]);
a = atof(argv[2]);
b = atof(argv[3]);
n = atoi(argv[4]);
}
printf("serial result = %.6f\n", trap_serial(a, b, n));
global_result = 0.0;
#pragma omp parallel num_threads(thread_count) reduction(+: global_result)
global_result += trap(a, b, n);
printf("threads= %d n= %d\n", thread_count, n);
printf("the integral from %.2f to %.2f = %.6f\n",
a, b, global_result);
return 0;
}
double f(double x) {
return x * x;
}
double trap_serial(double a, double b, int n) {
double h = (b-a) / n;
double res = (f(a) + f(b)) / 2;
for (int i = 1; i <= n-1; i++) {
res += f(a + i*h);
}
return res * h;
} 
double trap(double a, double b, int n) {
double  h, x, my_result;
double  local_a, local_b;
int  i, local_n;
int my_rank = omp_get_thread_num();
int thread_count = omp_get_num_threads();
h = (b-a) / n; 
local_n = n / thread_count;  
local_a = a + my_rank * local_n*h; 
local_b = local_a + local_n * h; 
my_result = (f(local_a) + f(local_b))/2.0; 
for (i = 1; i <= local_n-1; i++) {
x = local_a + i * h;
my_result += f(x);
}
my_result = my_result * h; 
return my_result; 
}
