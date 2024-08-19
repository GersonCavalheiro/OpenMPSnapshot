

#include<stdio.h>
#include<math.h>
#include"timer.h"
#include<omp.h>

#define N 100
#define NUM_THREADS 4
#define CHUNK 1

double a=0.0;
double b=1.0;


double fcalc(double x){
double f;
f = sqrt(1-x*x);
return f;
}

int main(){
double dx,x,fsum,pi;
int k,tid;

fsum = 0;
dx = (b-a)/N;

timespec before,after;
get_time(&before);

omp_set_num_threads(NUM_THREADS);
#pragma omp parallel shared(dx) private(x,k) reduction(+:fsum)
{
tid = omp_get_thread_num();
printf("%d \n", tid);
#pragma omp for schedule(dynamic,CHUNK)
for(k=1;k<N;k++){
x = a+(k-0.5)*dx;
fsum = fsum+(dx*fcalc(x));
}
}
pi = 4 * fsum;

get_time(&after);

timespec time_diff;

diff(&before,&after,&time_diff);
double time_s=time_diff.tv_sec + (double)(time_diff.tv_nsec)/1.0e9;
printf("time = %.09lf\n", time_s);

printf("pi: %.03lf\n",pi);
return 0;
}

