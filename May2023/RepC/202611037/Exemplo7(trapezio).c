#include <stdio.h>
#include <math.h>
#include <omp.h>
double f(double x) {
double return_val;
return_val = exp(x);
return return_val;
}
void main() {
double integral;     
double a, b;            
long  i,n;                  
double h;                
double x;
a = 0.0;
b = 1.0;
n = 800000000;
h = (b-a)/n;
integral = (f(a) + f(b))/2.0;
omp_set_num_threads(8);
double start = omp_get_wtime();
#pragma omp parallel for reduction(+:integral) shared(n, h)
for (i = 1; i < n-1; i++) {
integral += f(a + i*h);
}
integral *= h ;
printf("Com n = %ld trapezoides, a estimativa \n", n);
printf("da integral de %f ate %f = %lf \n", a, b, integral);
printf("Tempo: \t %f \n", omp_get_wtime()-start);
}