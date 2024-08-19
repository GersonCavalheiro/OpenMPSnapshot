#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
void Usage(char* prog_name);
double f(double x);    
double Trap(double a, double b, int n, int thread_count);
void Print_iters(int iters[], int n);
int main(int argc, char* argv[]) {
double  global_result = 0.0;  
double  a, b;                 
int     n;                    
int     thread_count;
if (argc != 2) Usage(argv[0]);
thread_count = strtol(argv[1], NULL, 10);
printf("Enter a, b, and n\n");
scanf("%lf %lf %d", &a, &b, &n);
global_result = Trap(a, b, n, thread_count);
printf("With n = %d trapezoids, our estimate\n", n);
printf("of the integral from %f to %f = %.14e\n",
a, b, global_result);
return 0;
}  
void Usage(char* prog_name) {
fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
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
int* iters = malloc(n*sizeof(int));
h = (b-a)/n;
approx = (f(a) + f(b))/2.0;
#pragma omp parallel for num_threads(thread_count) reduction(+: approx) schedule(static)
for (i = 1; i <= n-1; i++) {
approx += f(a + i*h);
iters[i] = omp_get_thread_num();
}
approx = h*approx;
Print_iters(iters, n);
free(iters);
return approx;
}  
void Print_iters(int iters[], int n) {
int i, curr_thread;
printf("iters = ");
for (i = 0; i < n; i++)
printf("%d ", iters[i]);
printf("\n\n");
printf("Thread\tIterations\n");
printf("  %d  \t  0 -- ", iters[0]);
curr_thread = iters[0];
for (i = 1; i < n-1; i++)
if (curr_thread != iters[i]) {
printf("%d\n", i-1);
printf("  %d  \t  %d -- ", iters[i], i);
curr_thread = iters[i];
}
if (curr_thread == iters[n-1]) {
printf("%d\n", n-1);
} else {
printf("%d\n", n-2);
printf("  %d  \t  %d -- %d\n", iters[n-1], n-1, n-1);
}
printf("\n");
}  
