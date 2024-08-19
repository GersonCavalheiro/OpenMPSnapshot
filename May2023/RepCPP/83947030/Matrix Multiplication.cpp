#include <stdio.h>
#include <omp.h>
#define SIZE 4000

double A[SIZE][SIZE];
double B[SIZE][SIZE];
double C[SIZE][SIZE]={{0}};

int main()
{
int t, u;

for (t = 0; t < SIZE; t++) {
for (u = 0; u < SIZE; u++) {
A[t][u] = B[u][t] = 1;
}
}

double start_time = omp_get_wtime();
int tid;
#pragma omp parallel num_threads(SIZE) private(tid)
{
tid=omp_get_thread_num();
for(int i=0;i<SIZE;i++)
for(int j=0;j<SIZE;j++)
C[i][j]=C[i][j]+A[i][tid]*B[tid][j];	
}							                    

double elapsed = omp_get_wtime()-start_time;

fprintf(stderr, "Time: %f s\n", elapsed);
}

