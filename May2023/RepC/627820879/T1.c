#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#define MIN(a,b) ((a) < (b) ? a : b)
void get_time(struct timespec* t) {
clock_gettime(CLOCK_MONOTONIC, t);
}
void get_clockres(struct timespec* t) {
clock_getres(CLOCK_MONOTONIC, t);
}
void multiply(int n, double * a, double * b, double * c);
void blockMultiply(int n, int bn, double * a, double * b, double * c);
void fillMatrix(int n, double * matrix);
void printMatrixByRows(int n, double * matrix);
void printMatrixByRowsInFile(int n, double * matrix, char filename[]);
double * createMatrix(int n);
int main(int argc, char * argv[]) {
unsigned int mSize = 0, opt = 0, runs, i;
struct timespec t1, t2, dt;
double time, flops, gFlops;
double * a, * b, * c;
if (argc == 2 && isdigit(argv[1][0])) {
mSize = atoi(argv[1]);
}
else if(argc == 3 && isdigit(argv[1][0]) && isdigit(argv[2][0]))
{
mSize = atoi(argv[1]);
opt   = atoi(argv[2]);
}else {
printf("USAGE\n   %s [SIZE] [opt]\n", argv[0]);
return 0;
}
get_clockres(&t1);
printf("Timer resolution is %lu nano seconds.\n",t1.tv_nsec);
a = (double*)createMatrix(mSize);
b = (double*)createMatrix(mSize);
c = (double*)createMatrix(mSize);
fillMatrix(mSize, a);
fillMatrix(mSize, b);
flops = (double)mSize * (double)mSize * (double)mSize * 2.0;
printf("Starting benchmark with mSize = %d and opt = %d.\n",mSize,opt);
runs = time = 0;
while (runs < 5) {
for (i = 0; i < mSize*mSize; i++) {
c[i] = 0;
}
get_time(&t1);
if (opt == 0)
multiply(mSize, a, b, c);
else
{
blockMultiply(mSize, opt, a, b, c);
}
get_time(&t2);
if ((t2.tv_nsec - t1.tv_nsec) < 0) {
dt.tv_sec = t2.tv_sec - t1.tv_sec - 1;
dt.tv_nsec = 1000000000 - t1.tv_nsec + t2.tv_nsec;
}else {
dt.tv_sec = t2.tv_sec - t1.tv_sec;
dt.tv_nsec = t2.tv_nsec - t1.tv_nsec;
}
time += dt.tv_sec + (double)(dt.tv_nsec)*0.000000001;
runs ++;
}
gFlops = ((flops/1073741824.0)/time)*runs;
printf("MATRIX SIZE: %i, GFLOPS: %f, RUNS: %i\n",mSize, gFlops, runs);
printf ("Mean execution time: %f\n", (time/runs));
free(a);
free(b);
free(c);
}
void multiply(int n, double * a, double * b, double * c) {
int i, j, k;
#pragma omp parallel for collapse(2) shared(a,b,c) private(i,j,k)
for (i = 0; i < n; i++) {
for (j = 0; j < n; j++) {
double sum = 0.0;
#pragma omp simd reduction(+:sum)
for (k = 0; k < n; k++) {
sum += a[i*n+k] * b[k*n+j];
}
c[i*n+j] = sum;
}
}
}
void blockMultiply(int n, int bn, double * a, double * b, double * c) {
int i, j, k, ii, jj;
#pragma omp parallel for collapse(2) shared(a,b,c) private(i,j,k,ii,jj)
for (ii = 0; ii < n; ii +=bn)
for (jj = 0; jj < n; jj += bn)
for (i = ii; i < MIN(ii+bn,n); i++)
for (j = jj; j < MIN(jj+bn,n); j++) {
double sum = 0.0;
#pragma omp simd reduction(+:sum)
for (k = 0; k < n; k++) {
sum += a[i*n+k] * b[k*n+j];
}
c[i*n+j] = sum;
}
}
double * createMatrix(int n) {
int i;
double * matrix = (double*) calloc(n*n,sizeof(double));
return matrix;
}
void fillMatrix(int n, double * matrix) {
int i;
for (i = 0; i < n*n; i++) {
matrix[i] = (rand()%10) - 5; 
}
}
void printMatrixByRows(int n, double * matrix) {
int i, j;
printf("{");
for (i = 0; i < n; i++) {
printf("[");
for (j = 0; j < n; j++) {
printf("%d",(int)matrix[i*n+j]);
if (j != n-1)
printf(",");
else
printf("]");
}
if (i != n-1)
printf(",\n");
}
printf("}\n");
}
void printMatrixByRowsInFile(int n, double *matrix, char filename[]) {
int i, j;
FILE *fp = fopen(filename, "w");
fprintf(fp, "{");
for (i = 0; i < n; i++) {
fprintf(fp, "[");
for (j = 0; j < n; j++) {
fprintf(fp, "%d",(int)matrix[i*n+j]);
if (j != n-1)
fprintf(fp, ",");
else
fprintf(fp, "]");
}
if (i != n-1)
fprintf(fp, ",\n");
}
fprintf(fp, "}\n");
fclose(fp);
}
