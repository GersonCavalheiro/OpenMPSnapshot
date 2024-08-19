#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#define MROWS 1024
#define MCOLS 1024
double max(double a, double b);
double **createGrid(int m, int n);
double **initGridSerial(int m, int n, double l, double r, double u, double d);
double **initGridParallel(int m, int n, double l, double r, double u, double d);
double **freeGrid(double **grid, int m, int n);
void displayGrid(double **grid, int m, int n);
double gaussSeidelSerial(double **grid, int m, int n, double eps, int *iterations, int maxIterations, double *wtime);
double gaussSeidelSerialIterations(double **grid, int m, int n, double eps, int *iterations, int maxIterations);
double gaussSeidelRBSerial(double **grid, int m, int n, double eps, int *iterations, int maxIterations, double *wtime);
double gaussSeidelRBSerialIterations(double **grid, int m, int n, double eps, int *iterations, int maxIterations);
double gaussSeidelRBParallel(double **grid, int m, int n, double eps, int *iterations, int maxIterations, double *wtime);
double gaussSeidelRBParallelIterations(double **grid, int m, int n, double eps, int *iterations, int maxIterations);
int main(int argc, char*argv[]) {
int m, n, p;
double eps = 0.001;
double l = 100, r = 100, u = 100, d = 0;
double **tgrid;
double err;
int iterations;
int maxIterations = 100000;
double algtime, e2etime;
int gss = 0;
int rbs = 0;
int rbp = 0;
if(argc!=5) {
printf("Provide m, n, method, p from Command Line\n");
return(0);
}
else {
m = atoi(argv[1]);
n = atoi(argv[2]);
p = atoi(argv[4]);
if(strcmp(argv[3], "gss")==0)
gss = 1;
else if(strcmp(argv[3], "rbs")==0)
rbs = 1;
else if(strcmp(argv[3], "rbp")==0)
rbp = 1;
else {
printf("Incorrect method chosen\n");
return(0);
}
}
if(m>MROWS || m<=0) {
printf("m should be between 0 and %d\n", MROWS);
return(0);
}
if(n>MCOLS || n<=0) {
printf("n should be between 0 and %d\n", MCOLS);
return(0);
}
if(eps<=0) {
printf("eps should be greater than 0\n");
return(0);
}
if(rbp==1)
omp_set_num_threads(p);
e2etime = omp_get_wtime();
if(gss==1 || rbs==1)
tgrid = initGridSerial(m, n, l, r, u, d);
else
tgrid = initGridParallel(m, n, l, r, u, d);
if(tgrid == NULL) {
printf("tgrid could not be initialized\n");
return(0);
}
if(gss == 1)
err = gaussSeidelSerial(tgrid, m, n, eps, &iterations, maxIterations, &algtime);
else if(rbs == 1)
err = gaussSeidelRBSerial(tgrid, m, n, eps, &iterations, maxIterations, &algtime);
else if(rbp == 1)
err = gaussSeidelRBParallel(tgrid, m, n, eps, &iterations, maxIterations, &algtime);
e2etime = omp_get_wtime() - e2etime;
printf("%s,%d,%d,%d,%lf,%d,%lf,%lf\n", argv[3], omp_get_max_threads(), m, n, err, iterations, algtime, e2etime);
tgrid = freeGrid(tgrid, m, n);
return(0);
}
double max(double a, double b) {
if(a>b)
return a;
else
return b;
}
double **createGrid(int m, int n) {
double **grid;
int i;
grid = (double**) malloc(m*sizeof(double*));
for(i=0;i<m;i++)
grid[i] = (double*) malloc(n*sizeof(double));
return(grid);
}
double **initGridSerial(int m, int n, double l, double r, double u, double d) {
double **grid;
grid = createGrid(m, n);
int i,j;
if(grid == NULL)
return(NULL);
for(i=0; i<n;i++) {
grid[0][i] = d;
grid[m-1][i] = u;
}
for(j=1; j<m-1; j++) {
grid[j][0] = l;
grid[j][n-1] = r;
}
double mean = (((double)l+r)*(m-2) + ((double)u+d)*(n))/(2*n+2*m-4);
for(j=1; j<m-1; j++)
for(i=1; i<n-1; i++)
grid[j][i]=mean;
return grid;
}
double **initGridParallel(int m, int n, double l, double r, double u, double d) {
double **grid;
grid = createGrid(m, n);
int i,j;
if(grid == NULL)
return(NULL);
for(i=0; i<n;i++) {
grid[0][i] = d;
grid[m-1][i] = u;
}
for(j=1; j<m-1; j++) {
grid[j][0] = l;
grid[j][n-1] = r;
}
double mean = (((double)l+r)*(m-2) + ((double)u+d)*(n))/(2*n+2*m-4);
#pragma omp parallel for private(i)
for(j=1; j<m-1; j++) {
for(i=1; i<n-1; i++)
grid[j][i]=mean;
}
return grid;
}
double **freeGrid(double **grid, int m, int n) {
int i;
for(i=0; i<m; i++)
free(grid[i]);
free(grid);
return NULL;
}
void displayGrid(double **grid, int m, int n) {
printf("\n");
int i,j;
for(j=m-1; j>=0; j--) {
for(i=0; i<n; i++)
printf("%lf ", grid[j][i]);
printf("\n");
}
}
double gaussSeidelSerial(double **grid, int m, int n, double eps, int *iterations, int maxIterations, double *wtime) {
double err;
*iterations = 0;
*wtime = omp_get_wtime();
err = gaussSeidelSerialIterations(grid, m, n, eps, iterations, maxIterations);
*wtime = omp_get_wtime() - *wtime;
return(err);
}
double gaussSeidelSerialIterations(double **grid, int m, int n, double eps, int *iterations, int maxIterations) {
int i,j;
double newVal;
double error = 10*eps;
while(error>eps && *iterations<maxIterations) {
error = 0;
for(j=1; j<m-1; j++) {
for(i=1; i<n-1; i++) {
newVal = 0.25*(grid[j-1][i]+grid[j+1][i]+grid[j][i-1]+grid[j][i+1]);
error = max(error, fabs(newVal - grid[j][i]));
grid[j][i] = newVal;
}
}
*iterations = *iterations+1;
}
return(error);
}
double gaussSeidelRBSerial(double **grid, int m, int n, double eps, int *iterations, int maxIterations, double *wtime) {
double err;
*iterations = 0;
*wtime = omp_get_wtime();
err = gaussSeidelRBSerialIterations(grid, m, n, eps, iterations, maxIterations);
*wtime = omp_get_wtime() - *wtime;
return(err);
}
double gaussSeidelRBSerialIterations(double **grid, int m, int n, double eps, int *iterations, int maxIterations) {
int i,j;
double newVal;
double error = 10*eps;
while(error>eps && *iterations<maxIterations) {
error = 0;
for(j=1; j<m-1; j++) {
for(i=1; i<n-1; i++)
if((i+j)%2==0) {
newVal = 0.25*(grid[j-1][i]+grid[j+1][i]+grid[j][i-1]+grid[j][i+1]);
error = max(error, fabs(newVal - grid[j][i]));
grid[j][i] = newVal;
}
}
for(j=1; j<m-1; j++) {
for(i=1; i<n-1; i++)
if((i+j)%2==1) {
newVal = 0.25*(grid[j-1][i]+grid[j+1][i]+grid[j][i-1]+grid[j][i+1]);
error = max(error, fabs(newVal - grid[j][i]));
grid[j][i] = newVal;
}
}
*iterations = *iterations+1;
}
return(error);
}
double gaussSeidelRBParallel(double **grid, int m, int n, double eps, int *iterations, int maxIterations, double *wtime) {
double err;
*iterations = 0;
*wtime = omp_get_wtime();
err = gaussSeidelRBParallelIterations(grid, m, n, eps, iterations, maxIterations);
*wtime = omp_get_wtime() - *wtime;
return(err);
}
double gaussSeidelRBParallelIterations(double **grid, int m, int n, double eps, int *iterations, int maxIterations) {
int j;
double error = 10*eps;
while(error>eps && *iterations<maxIterations) {
double errorR = 0;
double errorB = 0;
#pragma omp parallel for reduction(max:errorR)
for(j=1; j<m-1; j++) {
int i;
double newVal;
for(i=1; i<n-1; i++)
if((i+j)%2==0) {
newVal = 0.25*(grid[j-1][i]+grid[j+1][i]+grid[j][i-1]+grid[j][i+1]);
errorR = max(errorR, fabs(newVal - grid[j][i]));
grid[j][i] = newVal;
}
}
#pragma omp parallel for reduction(max:errorB)
for(j=1; j<m-1; j++) {
int i;
double newVal;
for(i=1; i<n-1; i++)
if((i+j)%2==1) {
newVal = 0.25*(grid[j-1][i]+grid[j+1][i]+grid[j][i-1]+grid[j][i+1]);
errorB = max(errorB, fabs(newVal - grid[j][i]));
grid[j][i] = newVal;
}
}
error = max(errorR, errorB);
*iterations = *iterations+1;
}
return(error);
}
