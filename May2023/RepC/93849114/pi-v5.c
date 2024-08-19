#include <stdio.h>
#include <stdlib.h>
#include <omp.h>	
#if _DEBUG_
#define _DEBUG_ 1
#else
#define _DEBUG_ 0
#endif
int main(int argc, char *argv[]) {
double x, sum=0.0, pi=0.0;
#if !_DEBUG_
double start,end;
#endif
int i;
const char Usage[] = "Usage: pi <num_steps> (try 1000000000)\n";
if (argc < 2) {
fprintf(stderr, Usage);
exit(1);
}
int num_steps = atoi(argv[1]);
double step = 1.0/(double) num_steps;
#if !_DEBUG_
start= omp_get_wtime();
#endif
#pragma omp parallel private(i, x)
{
int id = omp_get_thread_num();
int num_threads = omp_get_num_threads();
for (i=id; i < num_steps; i=i+num_threads) {
x = (i+0.5)*step;
#pragma omp atomic
sum += 4.0/(1.0+x*x);
#if _DEBUG_
printf("thread id:%d it:%d\n",id,i);
#endif
}
}
pi = step * sum;
#if !_DEBUG_
end = omp_get_wtime();
printf("Wall clock execution time  = %.9f seconds\n", end-start);
#endif
printf("Value of pi = %12.10f\n", pi);
return EXIT_SUCCESS;
}
