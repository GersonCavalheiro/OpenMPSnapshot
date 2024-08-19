#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#define m 2000
#define n 3000
#define o 2000
double **mat1, **mat2, **result, **trans;
int thread_count;
int thread_num;
double sum;
int main(int argc, char *argv[])
{
if (argc != 2) {
fprintf( stderr, "%s <number of threads>\n", argv[0] );
return -1;
}
thread_count = atoi( argv[1] );
int i, j, k;
struct timeval start, end;
omp_set_dynamic(0);
omp_set_num_threads(thread_count);
thread_num = omp_get_max_threads ( );
printf ( "\n" );
printf ( "  The number of threads used    = %d\n", thread_num );
mat1 = (double **) malloc(m * sizeof(double *));       
for (i=0;i<m;i++)                                      
mat1[i] = (double *) malloc(n * sizeof(double));
mat2 = (double **) malloc(n * sizeof(double *));       
for (i=0;i<n;i++)                                      
mat2[i] = (double *) malloc(o * sizeof(double));
result = (double **) malloc(m * sizeof(double *));     
for (i=0;i<m;i++)                                      
result[i] = (double *) malloc(o * sizeof(double));
trans = (double **) malloc(o * sizeof(double *));     
for (i=0;i<o;i++)                                     
trans[i] = (double *) malloc(m * sizeof(double));
srand(time(NULL));                                 
for (i = 0; i < m; i++) {
for (j = 0; j < n; j++) {
mat1[i][j] = (double) rand()/ RAND_MAX;      
}
}
for (i = 0; i < n; i++) {
for (j = 0; j < o; j++) {
mat2[i][j] = (double) rand()/ RAND_MAX;       
}
}
gettimeofday(&start, NULL);                         
#pragma omp parallel  num_threads(thread_num) reduction(+: sum) private(i,j,k) shared(mat1,mat2)
{
#pragma omp for
for (i = 0; i < m; i++) {
for (j = 0; j < o; j++) {
result[i][j] = 0;
sum = 0;
for (k =0; k < n; k++) {
sum+= mat1[i][k] * mat2[k][j];
}
result[i][j] =sum;
}
}
#pragma omp for
for (i = 0; i < m; i++) {
for (j = 0; j < o; j++) {
trans[j][i] = result[i][j];
}
}
}
gettimeofday(&end, NULL);    
printf("\n Execution Time: %fs \n", ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0));
for (i = 0; i < m; i++){
free(mat1[i]);
}
free(mat1);
for (i = 0; i < n; i++){
free(mat2[i]);
}
free(mat2);
for (i = 0; i < m; i++){
free(result[i]);
}
free(result);
for (i = 0; i < o; i++){
free(trans[i]);
}
free(trans);
return 0;
}
