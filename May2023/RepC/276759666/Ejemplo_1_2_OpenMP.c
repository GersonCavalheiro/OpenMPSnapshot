#include <omp.h> 
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
double f1(double *y,double x){
*y = 4*sin(x)+6;
}
double f2(double *y,double x){
*y = 8*exp(-2*x);
}
double f3(double *y,double x){
*y = log10(x)+cos(x);
}
double f4(double *y,double x){
*y = 3*pow(x,6)+6*pow(x,5);
}
int main(int argc, char* argv[]){
double a,b,c,d,y;
#pragma omp parallel sections
{ 
#pragma omp section
f1(&a,27.5);
#pragma omp section
f2(&b,0.43);
#pragma omp section
f3(&c,16.3);
#pragma omp section
f4(&d,2.3);
}
y = (a*b+c)*d;
printf("Resultado: %f\n",y);
return 0;
} 
