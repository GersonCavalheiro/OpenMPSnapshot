#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define RADIUS 1
int N_THREADS = 2;
long N_POINTS = 100;
int argparse(int argc, char *argv[]) {
for (long i = 0; i < argc; i++)
if (argv[i][0] == '-')
switch (argv[i][1]) {
case 't':
N_THREADS = atol(argv[++i]);
break;
case 'p':
N_POINTS = atol(argv[++i]);
break;
default:
fprintf(stderr, "Usage: %s [-option] [number]\n", argv[0]);
fprintf(stderr, "Option: -p [number-of-points]\n");
fprintf(stderr, "Option: -t [number-of-threads]\n");
exit(EXIT_FAILURE);
}
return 0;
}
int main(int argc, char *argv[]) {
argparse(argc, argv);
double M = 0;
#pragma omp parallel num_threads(N_THREADS) shared(M) firstprivate(N_POINTS) default(none)
{
unsigned int seed = time(NULL);
double x, y;
#pragma omp for private(x, y) reduction(+:M)
for(long i = 0; i < N_POINTS; i++){
x = (double) rand_r(&seed) / ((double) RAND_MAX / RADIUS);
y = (double) rand_r(&seed) / ((double) RAND_MAX / RADIUS);
if (x * x + y * y <= 1) M++;
}
}
printf("Pi: %f\n", M / (double) N_POINTS * 4.0);
return 0;
}
