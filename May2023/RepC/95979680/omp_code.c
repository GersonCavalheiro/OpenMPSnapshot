#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
void Usage(char* prog_name);
double f(double x);    
double Trap(double a, double b, int n, int thread_count);
int main(int argc, char* argv[]) {
double  global_result = 0.0;  
double  a, b;                 
int     n;                    
int     thread_count;
if (argc != 5) Usage(argv[0]);
thread_count = strtol(argv[1], NULL, 10);
a = strtod(argv[2],NULL);
b = strtod(argv[3],NULL);
n = strtol(argv[4],NULL,10);
global_result = Trap(a, b, n, thread_count);
printf("With n = %d trapezoids, our estimate\n", n);
printf("of the integral from %f to %f = %.14e\n",
a, b, global_result);
return 0;
}  
void Usage(char* prog_name) {
fprintf(stderr, "Usage: %s <number of threads> <a> <b> <n>\n", prog_name);
exit(0);
}  
double f(double x) {
double return_val;
return_val = x*x;
return return_val;
}  
double Trap(double a, double b, int n, int thread_count) {
double  h, approx;
int  i;
int *iterations;
iterations = (int*)malloc(sizeof(int)*n);
h = (b-a)/n; 
approx = (f(a) + f(b))/2.0; 
iterations[0] = 0;
#pragma omp parallel for num_threads(thread_count) reduction(+: approx) schedule(runtime)
for (i = 1; i < n; i++)
{
iterations[i] = omp_get_thread_num();
approx += f(a + i*h);
}
iterations[n-1] = 0;
approx = h*approx; 
printf("\nPrinting array of iterations:\n");
for (i = 1; i < n-1; i++)
printf("Iteration %d >> Thread %d\n", i, iterations[i]);
printf("\n");
free(iterations);
return approx;
}  
