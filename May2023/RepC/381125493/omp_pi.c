#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define R 1.0
double get_random() {
return (double) rand() / (double) RAND_MAX;
}
double omp_pi(int num_trials) {
double pi, x, y;
int num_points_circle = 0;
int i;
srand(time(NULL)); 
#pragma omp parallel for private(i) firstprivate(x,y) reduction(+:num_points_circle)
{
for(i = 0; i < num_trials; ++i) {
x = get_random();
y = get_random();
if(x*x+y*y <= R)
num_points_circle++;
}
}
pi = 4.0*(num_points_circle/(double) num_trials);
return pi;
}
int main(int argc, char* argv[]) {
const int num_trials = atoi(argv[1]);
double start_time = omp_get_wtime();
double omp_rs = omp_pi(num_trials);
double time_taken = omp_get_wtime() - start_time;
printf("PI = %.8f -- Time taken in PARALLEL-OMP: %.8fs.\n", omp_rs, time_taken);
return 0;
}
