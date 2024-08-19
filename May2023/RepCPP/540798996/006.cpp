
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <conio.h>
using namespace std;

int main(){

int n = 1000000000; 

float h,S,x;
int i, a, b;
int chunk = 10;
int count_treads = 50;

a = 1;
b = 5;
h=(b-a) * 1.0 / n;
S=0.000000;
float result=0.0000000;

double time_spent = 0.00000000;
double time_spent_par = 0.00000000;

clock_t begin =  clock();
for(i=0;i<n;++i)
{
x=a+i*h;
S=S+(1+sin(x)/(1+cos(x)));
}
S=h*S;
clock_t end =  clock();
time_spent += (double)(end - begin) / (CLOCKS_PER_SEC);
printf("\nSequential work time is %.10f seconds", time_spent);

S=0.0;
clock_t begin_par =  clock();
#pragma omp parallel for num_threads(count_treads) reduction(+:S)
for(i=0;i<n;++i)
{
x=a+i*h;
S=S+(1+sin(x)/(1+cos(x)));
}
S=h*S;
clock_t end_par =  clock();
time_spent_par += (double)(end_par - begin_par) / (CLOCKS_PER_SEC);
printf("\nParallel work time is %.10f seconds", time_spent_par);

return 0;
}