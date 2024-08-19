#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  N   ((1 << 6) + 2)
double   maxeps = 0.1e-7;
#define  itmax 100
double A [N][N][N];
double it_eps[itmax];
int end = 0;
int working_iterations[itmax + 1]; 
int access_to_matrix[itmax][N]; 
double relax(int);
void init();
void verify(); 
int main(int an, char **as)
{
(void) an;
(void) as;
double time_begin = omp_get_wtime();
const int MAX_THREADS_NUM = strtol(as[1], NULL, 10);
int i;
for (int i = 0; i < N; i++) {
access_to_matrix[0][i] = 1;
}
omp_set_num_threads(MAX_THREADS_NUM);
init();
#pragma omp parallel
{
while (!end) {
int it = 0;
#pragma omp critical
{
for (; it < itmax; it++) {
if (!working_iterations[it]) break;
}
if (it < itmax) {
working_iterations[it] = 1; 
}
} 
if (it < itmax && !end) {
double eps = relax(it);
it_eps[it] = eps;
if (eps < maxeps) end = 1;
} else {
end = 1;
break;
}
}
} 
for (i = 0; i < itmax; i++) {
printf("it=%4i  eps=%f\n", i + 1, it_eps[i]);
}
verify();
double time_end = omp_get_wtime();
fprintf(stderr, "time = %lf\n", time_end - time_begin);
return 0;
}
void init_part(double matrix[N][N], int size, int i_it)
{
int j_it, k_it;
for (j_it = 0; j_it < size; j_it++) {
for (k_it = 0; k_it < size; k_it++) {
if (i_it == 0 || i_it == N - 1 || j_it == 0 || j_it == N - 1 || k_it == 0 || k_it == N - 1) {
matrix[j_it][k_it] = 0.;
} else {
matrix[j_it][k_it] = (4. + i_it + j_it + k_it);
}
}
}
}
void init()
{
#pragma omp parallel
{
int i_it;
#pragma omp for 
for (i_it = 0; i_it < N; i_it++) {
init_part(A[i_it], N, i_it);
}
} 
}
double relax(int iteration)
{
double rel_eps = 0.;
int i, j, k;
for(i=1; i<=N-2; i++) {
while (!access_to_matrix[iteration][i]); 
for(j=1; j<=N-2; j++) {
for(k=1; k<=N-2; k++) {
float e;
e=A[i][j][k];
A[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1])/6.;
rel_eps=Max(rel_eps, fabs(e-A[i][j][k]));
}
}
access_to_matrix[iteration + 1][i - 1] = 1;
}  
access_to_matrix[iteration + 1][i - 1] = 1; 
return rel_eps;
}
void verify()
{ 
float s;
s=0.;
int i, j, k;
for(i=0; i<=N-1; i++)
for(j=0; j<=N-1; j++)
for(k=0; k<=N-1; k++)
{
s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
}
printf("  S = %f\n",s);
}
