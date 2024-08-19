#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#define SEED 0
#define n 5000000
#define SAVE 1
struct timeval tv; 
double get_clock() {
struct timeval tv; int ok;
ok = gettimeofday(&tv, (void *) 0);
if (ok<0) { printf("gettimeofday error");  }
return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6); 
}
int main(int argc, char *argv[]) {
int i;
double *x,*y,*z;
double t1,t2;
int numthreads,tid;
#pragma omp parallel
{
numthreads = omp_get_num_threads();
tid = omp_get_thread_num();
if(tid==0)
printf("Running addition with %d threads\n",numthreads);
}
x = (double*) malloc(sizeof(double)*n);
y = (double*) malloc(sizeof(double)*n);
z = (double*) malloc(sizeof(double)*n);
srand(SEED); 
for(i=0;i<n;i++) {
x[i] = rand()%1000;
y[i] = rand()%1000;
}
t1 = get_clock();
#pragma omp parallel default(none) shared(z,x,y) private(i)
{
#pragma omp for schedule(static) 
for(i=0;i<n;i++) {
z[i] = x[i] + y[i];
}
}
t2 = get_clock();
printf("Time: %lf\n",(t2-t1));
if(SAVE) {
char outfile[100];
sprintf(outfile,"addition_out_%d.txt",numthreads);
printf("Outputting solution to %s\n",outfile);
FILE *fp = fopen(outfile,"w");
for(i=0; i<n; i++)
fprintf(fp,"%lf\n",z[i]);
fclose(fp);
}
free(x);
free(y);
free(z);
return 0;
}
