#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#define N 8192

double rtclock() {
struct timezone Tzp;
struct timeval Tp; 
int stat;
stat = gettimeofday (&Tp, &Tzp);
if (stat != 0) printf("Error return from gettimeofday: %d",stat);
return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

float A[N][N], B[N][N], C[N][N];


void myMult() {
#pragma omp target map(to:A, B) map(tofrom:C)
#pragma omp parallel for
for(int i = 0; i < N; ++i)
for(int k = 0; k < N; ++k)
for(int j = 0; j < N; ++j)
C[i][j] = A[i][k] * B[k][j];
}

int main(int argc, char *argv[]) {
if(argc != 4) {
fprintf(stderr, "Use: %s size nThreads nIter\n", argv[0]);
return -1;
}
int i, j, k, nt;
int nThreads = atoi(argv[2]);
int nIter = atoi(argv[3]);
omp_set_num_threads(nThreads);
memset(A, 0, N * N * sizeof(float));
memset(B, 0, N * N * sizeof(float));
memset(C, 0, N * N * sizeof(float));
printf("Initializing input matrices...\n");
for(i = 0; i < N; ++i) {
for(j = 0; j < N; ++j) {
A[i][j] = 1.0f;
B[i][j] = 1.0f;
C[i][j] = 0.0f;
}
}
printf("warm up run to overcome setup overhead\n");
myMult();
double aveTime, minTime=1e6, maxTime=0.0f;
printf("run the matrix multiplication function %d times\n", nIter);
for(i=0; i < nIter; i++) {
double startTime = rtclock();
myMult();
double endTime = rtclock();
double runtime = endTime - startTime;
maxTime=(maxTime > runtime)?maxTime:runtime;
minTime=(minTime < runtime)?minTime:runtime;
aveTime += runtime;
printf("Iteration %d: runtime %.3f\n", i, runtime);
}
aveTime /= nIter;
printf("maxRT %g minRT %g aveRT %g GFlop/s %g\n", maxTime, minTime, aveTime, 2e-9*N*N*N/aveTime);
return 0;
}

