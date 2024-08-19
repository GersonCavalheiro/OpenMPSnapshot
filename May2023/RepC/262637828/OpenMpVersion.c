#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include<stdbool.h>
# define NPOINTS 500
# define MAXITER 500
struct complex {
double real;
double imag;
};
int main() {
double start, finish;
double area, error, ztemp;
int numoutside = 0;
struct complex z, c;
bool stop = false;
start = omp_get_wtime();
#pragma omp parallel for private(ztemp, c, z, stop) shared(numoutside)
for (int i=0; i<NPOINTS; i++) {
#pragma omp parallel for private(ztemp, c, z, stop) shared(numoutside)
for (int j=0; j<NPOINTS; j++) {
c.real = -2.0+2.5*(double)(i)/(double)(NPOINTS)+1.0e-7;
c.imag = 1.125*(double)(j)/(double)(NPOINTS)+1.0e-7;
z=c;
for (int iter=0; iter<MAXITER && !stop; iter++) {
ztemp=(z.real*z.real)-(z.imag*z.imag)+c.real;
z.imag=z.real*z.imag*2+c.imag;
z.real=ztemp;
if ((z.real*z.real+z.imag*z.imag)>4.0e0) {
#pragma omp atomic
numoutside++;
stop = true;
}
}
stop = false;
}
}
finish = omp_get_wtime();
area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
error=area/(double)NPOINTS;
printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
printf("Time = %12.8f seconds\n",finish-start);
}
