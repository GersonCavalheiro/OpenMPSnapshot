#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <math.h>

double value_at(double x)
{
return sin(x);
}

double montecarlo(double x0, double xn, double y0, double yn ,int n,int nthreads,double rectArea)
{

double s = 0; 
double x,y;
int i;
#pragma omp parallel for schedule(dynamic,nthreads)  default(none) private(i) shared(n,x,y,x0,y0,yn,xn,nthreads)  reduction(+:s)
for (i = 0; i < n; i++) {

x = x0 + (xn-x0)*((double) rand() / (RAND_MAX)); 
y = y0 + (yn-y0)*((double) rand() / (RAND_MAX));

if (abs(y) <= abs(value_at(x)) ){
if (value_at(x)>0 && y >0 && y<=value_at(x) ){
s =s+1;
} 
if(value_at(x) < 0 && y < 0 && y>=value_at(x) )	{
s=s-1;
}
}

}

return ((rectArea*s)/n);
}

int main()
{   

double x0 = 0;
double xn = 3.14159;
double x,y;
int n = 1500;
int nthreads= 8;
omp_set_num_threads(nthreads);

double f = (xn-x0)/n;
double y0 = 0;
double yn = 1;


double rectArea = (xn-x0)*(yn-y0);
printf("Value of integral is %f\n",
(montecarlo(x0,xn,y0,yn,n,nthreads,rectArea)));

return 0;
}