#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#define n 1000
#define MIN(a,b) (((a)<(b))?(a):(b))
#define bsize 100
#define SEED 0
#define SAVE 1
struct timeval tv; 
double get_clock() {
struct timeval tv; int ok;
ok = gettimeofday(&tv, (void *) 0);
if (ok<0) { printf("gettimeofday error");  }
return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6); 
}
double **create_matrix() {
int i,j;
double **a;
a = (double**) malloc(sizeof(double*)*n);
for (i=0;i<n;i++) {
a[i] = (double*) malloc(sizeof(double)*n);
}
srand(SEED);
for (i=0;i<n;i++) {
for (j=0;j<n;j++) {
a[i][j] = rand()%10;
}
}
return a;
}
void free_matrix(double **a) {
int i;
for (i=0;i<n;i++) {
free(a[i]);
}
free(a);
}
int main(int argc, char *argv[]) {
int i,j,k,ii,jj,kk;
double **A, **B, **C;
double t1,t2;
int numthreads,tid;
#pragma omp parallel
{
numthreads = omp_get_num_threads();
tid = omp_get_thread_num();
if(tid==0)
printf("Running tiled with %d threads\n",numthreads);
}
A = create_matrix();
B = create_matrix();
C = (double**) malloc(sizeof(double*)*n);
for (i=0;i<n;i++) {
C[i] = (double*) malloc(sizeof(double)*n);
}
for(i=0; i<n; i++) {
for(j=0; j<n; j++)
C[i][j] = 0.0;
}
t1 = get_clock();
#pragma omp parallel default(none) shared(A,B,C) private(ii,jj,kk,i,j,k)
{
#pragma omp for collapse(1) schedule(static)
for (ii = 0; ii<n; ii+=bsize) {
for (jj = 0; jj<n; jj+=bsize) {
for(kk = 0; kk<n; kk+=bsize) {
for (i = ii; i < MIN(ii+bsize,n); i++) {
for (j = jj; j < MIN(jj+bsize,n); j++) {
for (k = kk; k < MIN(kk+bsize,n); k++) {
C[i][j] += A[i][k] * B[k][j];
}
}
}
}
}
}
}
t2 = get_clock();
printf("Time: %lf\n",(t2-t1));
if(SAVE) {
char outfile[100];
sprintf(outfile,"tiled_out_%d.txt",numthreads);
printf("Outputting solution to %s\n",outfile);
FILE *fp = fopen(outfile,"w");
for(i=0; i<n; i++) {
for(j=0; j<n; j++) {
fprintf(fp,"%lf\n",C[i][j]);
}
}
fclose(fp);
}
free_matrix(A);
free_matrix(B);
free_matrix(C);
return 0;
}
