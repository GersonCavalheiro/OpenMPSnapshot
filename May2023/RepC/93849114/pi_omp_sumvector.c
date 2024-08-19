#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>	
double getusec_() {
struct timeval time;
gettimeofday(&time, NULL);
return ((double)time.tv_sec * (double)1e6 + (double)time.tv_usec);
}
#define START_COUNT_TIME stamp = getusec_();
#define STOP_COUNT_TIME(_m) stamp = getusec_() - stamp;\
stamp = stamp/1e6;\
printf ("%0.6f\n", stamp);
#define MAXTHREADS 16
double  sumvector[MAXTHREADS]; 
int main(int argc, char *argv[]) {
double stamp;
double x, sum=0.0, pi=0.0;
double step;
const char Usage[] = "Usage: pi <num_steps> <num_threads>\n";
if (argc < 3) {
fprintf(stderr, Usage);
exit(1);
}
long int num_steps = atoi(argv[1]);
step = 1.0/(double) num_steps;
int num_threads = atoi(argv[2]);
START_COUNT_TIME;
for (int i=0; i<num_threads; i++)
sumvector[i] = 0.0;
#pragma omp parallel private(x) num_threads(num_threads)
{
int myid = omp_get_thread_num();
#pragma omp for
for (long int i=0; i<num_steps; ++i) {
x = (i+0.5)*step;
sumvector[myid] += 4.0/(1.0+x*x);
}
}
for (int i=0; i<num_threads; i++)
sum += sumvector[i];
pi = step * sum;
STOP_COUNT_TIME("Total execution time");
return EXIT_SUCCESS;
}
