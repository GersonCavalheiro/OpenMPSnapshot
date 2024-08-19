#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#define SEED 0
#define n 5000
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
int i,j;
double **A;
double t1,t2,temp;
int numthreads,tid;
#pragma omp parallel
{
numthreads = omp_get_num_threads();
tid = omp_get_thread_num();
if(tid==0)
printf("Running transpose with %d threads\n",numthreads);
}
A = create_matrix();
t1 = get_clock();
#pragma omp parallel default(none) shared(A) private (j, temp,i) 
{
for(i=0;i<n;i++) {
#pragma omp for
for(j=i+1;j<n;j++) {
temp=A[i][j];
A[i][j] = A[j][i];
A[j][i] = temp;
}
}
}	
t2 = get_clock();
printf("Time: %lf\n",(t2-t1));
if(SAVE) {
char outfile[100];
sprintf(outfile,"transpose_out_%d.txt",numthreads);
printf("Outputting solution to %s\n",outfile);
FILE *fp = fopen(outfile,"w");
for(i=0; i<n; i++) {
for(j=0; j<n; j++) {
fprintf(fp,"%lf\n",A[i][j]);
}
}
fclose(fp);
}
free_matrix(A);
return 0;
}
