

#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <conio.h>
using namespace std;

int main(){

int n = 1000000; 

int i;

int *a = (int*) calloc(n, sizeof(int));
int *b = (int*) calloc(n, sizeof(int));
int *res = (int*) calloc(n, sizeof(int));
int *res_par = (int*) calloc(n, sizeof(int));

double time_spent = 0.00000000;
double time_spent_par = 0.00000000;

for(int i = 0; i < n; ++i) {
a[i] = rand();
b[i] = rand();
}

clock_t begin =  clock();
for(int i = 0; i < n; ++i)
{
res[i] = a[i]*b[i];
}
clock_t end =  clock();
time_spent += (double)(end - begin) / (CLOCKS_PER_SEC);
printf("\nSequential work time is %.10f seconds", time_spent);
free(res);

clock_t begin_par =  clock();
#pragma omp parallel shared(a,b,res_par,n) private(i)
{
#pragma omp sections nowait
{
#pragma omp section
for (i=0; i < n/2; ++i)
res_par[i] = a[i]*b[i];
#pragma omp section
for (i=n/2; i < n; ++i)
res_par[i] = a[i]*b[i];
}
} 
clock_t end_par =  clock();
time_spent_par += (double)(end_par - begin_par) / (CLOCKS_PER_SEC);
printf("\nParallel work time is %.10f seconds", time_spent_par);

free(a);
free(b);
free(res_par);
return 0;
}