#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define ll long long int
ll fibserial(int n){
if (n <= 1)
return n;
return fibserial(n-1) + fibserial(n-2);
}
ll comp_fib_numbers(int n, int threads)
{
long fnm1, fnm2, fn;                   
if (n <= 1) return(n);
if (threads == 1) return fibserial(n);
#pragma omp task shared(fnm1)
fnm1 = comp_fib_numbers(n-1, threads/2);
#pragma omp task shared(fnm2)
fnm2 = comp_fib_numbers(n-2, threads - threads/2);
#pragma omp taskwait
return(fnm1 + fnm2);
}
int main(int argc, char **argv)
{
int n;
double start_time, run_time;
ll result;
if(argc < 2){
printf("Usage ./a.out <number>\n");
exit(1);
}
n = atoi(argv[1]);
omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel
{
#pragma omp single nowait
{
start_time = omp_get_wtime();
result = comp_fib_numbers(n, omp_get_num_threads());
run_time = omp_get_wtime() - start_time;
printf("Ans: %lld  \n Time to compute(in parallel) : %f\n", result, run_time);
} 
}
start_time = omp_get_wtime();
result = fibserial(n);
run_time = omp_get_wtime() - start_time;
printf("Ans: %lld  \n Time to compute(in series) : %f\n", result, run_time);
return 0;
}
