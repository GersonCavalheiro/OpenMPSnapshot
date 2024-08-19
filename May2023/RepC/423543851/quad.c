#include <stdio.h>
#include <math.h>
#include <omp.h>
#define TOL 1e-8
#define THREADS 32
double func(double x) {
return sin(x*x);
}
double quad(double (*f)(double), double lower, double upper, double tol) {
double quad_res;        
double h;               
double middle;          
double quad_coarse;     
double quad_fine;       
double quad_lower;      
double quad_upper;      
double eps;             
h = upper - lower;
middle = (lower + upper) / 2;
quad_coarse = h * (f(lower) + f(upper)) / 2.0; 
quad_fine = h/2 * (f(lower) + f(middle)) / 2.0 + h/2 * (f(middle) + f(upper)) / 2.0; 
eps = fabs(quad_coarse - quad_fine);
if (eps > tol) {
#pragma omp task shared(quad_lower) final(h < 1.0)
quad_lower = quad(f, lower, middle, tol / 2);
quad_upper = quad(f, middle, upper, tol / 2);
#pragma omp taskwait
quad_res = quad_lower + quad_upper;
} else {
quad_res = quad_fine;
}
return quad_res;
}
int main(int argc, char* argv[]) {
double quadrature;
double dt = omp_get_wtime();
omp_set_num_threads(THREADS);
#pragma omp parallel
#pragma omp master
quadrature = quad(func, 0.0, 50.0, TOL);
dt = omp_get_wtime() - dt;
printf("Integral: %lf\nCas: %lf s\n", quadrature, dt);
return 0;
}
