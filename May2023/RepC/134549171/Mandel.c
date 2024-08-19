

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

# define NPOINTS 1000
# define MAXITER 1000


struct d_complex{
double r;
double i;
};

void testpoint(struct d_complex);
struct d_complex c;
int numoutside = 0;

int main (int argc, char *argv[])
{
unsigned int thread_qty = atoi(argv[1]);
if(thread_qty == 1){
int l;
omp_sched_t k;
omp_get_schedule(&k,&l);
printf("%d %d\n",k,l);
}

omp_set_num_threads(thread_qty);
double start_time, run_time;
int i,j;
int nthreads;
double area, error, eps  = 1.0e-5;

start_time = omp_get_wtime();

#pragma omp parallel default(shared)
{
#pragma omp single
nthreads = omp_get_num_threads();

#pragma omp for private(c,j) firstprivate(eps)
for (i=0; i<NPOINTS; i++) {
for (j=0; j<NPOINTS; j++) {
c.r = -2.0+2.5*(double)(i)/(double)(NPOINTS)+eps;
c.i = 1.125*(double)(j)/(double)(NPOINTS)+eps;
testpoint(c);
}
}
}


area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
error=area/(double)NPOINTS;
run_time = omp_get_wtime() - start_time;
printf("%lf\n",run_time);

}

void testpoint(struct d_complex c){

int iter;
double temp;
struct d_complex z;

z=c;
for (iter=0; iter<MAXITER; iter++){
temp = (z.r*z.r)-(z.i*z.i)+c.r;
z.i = z.r*z.i*2+c.i;
z.r = temp;
if ((z.r*z.r+z.i*z.i)>4.0) {
#pragma omp atomic
numoutside++;
break;
}
}

}

