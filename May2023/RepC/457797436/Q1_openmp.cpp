#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#define TOL 1e-5
#define ITER_LIMIT 1000
int P, n;
float **A, diff = 0.0;
void Initialize (float **X)
{
for (int i=0; i<n+2; i++) for (int j=0; j<n+2; j++) X[i][j] = ((float)(random() % 100)/100.0);
}
void Solve ()
{
int pid = omp_get_thread_num();
int done = 0, iters = 0;
float temp, local_diff;
while (!done) {
local_diff = 0.0;
#pragma omp master
diff = 0.0;
#pragma omp barrier
for (int i = pid*(n/P)+1; i < ((pid+1)*(n/P))+1; i++) {
for (int j = 1; j < n+1; j++) {
temp = A[i][j];
A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
local_diff += fabs(A[i][j] - temp);
}
}
#pragma omp atomic
diff += local_diff;
#pragma omp barrier
iters++;
if ((diff/(n*n) < TOL) || (iters == ITER_LIMIT)) done = 1;
#pragma omp barrier
}
printf("[%d] diff = %.10f\n", iters, diff/(n*n));
}
int main (int argc, char **argv)
{
struct timeval tv0, tv1;
struct timezone tz0, tz1;
char buffer[64];
if (argc != 3) {
printf("Need grid size (n) and number of threads (P).\nAborting...\n");
exit(1);
}
n = atoi(argv[1]);
P = atoi(argv[2]);
A = (float**)malloc((n+2)*sizeof(float*));
assert(A != NULL);
for (int i=0; i<n+2; i++) {
A[i] = (float*)malloc((n+2)*sizeof(float));
assert(A[i] != NULL);
}
Initialize(A);
gettimeofday(&tv0, &tz0);
#pragma omp parallel num_threads (P)
Solve();
gettimeofday(&tv1, &tz1);
printf("Time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
return 0;
}