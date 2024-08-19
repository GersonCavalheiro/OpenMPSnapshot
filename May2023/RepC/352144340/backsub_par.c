#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
void create_matrix(float *x, float *b, float **a, int N);
void calculate_x_matrix(float *x, float *b, float **a, int *flag, int N);
void print_results(float *x, int N);
void validate_results(float *x, float *b, float **a, int N);
void display_time(double start, double end);
void main ( int argc, char *argv[] )  {
int   i, j, N, *flag;
float *x, *b, **a, sum;
char any;
if (argc != 2) {
printf ("Usage : %s <matrix size>\n", argv[0]);
exit(1);
}
N = strtol(argv[1], NULL, 10);
a = (float **) malloc ( N * sizeof ( float *) );
for ( i = 0; i < N; i++) 
a[i] = ( float * ) malloc ( N * sizeof ( float ) );
b = ( float * ) malloc ( N * sizeof ( float ) );
x = ( float * ) malloc ( N * sizeof ( float ) );
flag = ( int *) malloc ( N * sizeof ( int ) );
create_matrix(x, b, a, N); 
double begin = omp_get_wtime();
calculate_x_matrix(x, b, a, flag, N);
double end = omp_get_wtime();
print_results(x, N);
display_time(begin, end);
}	
void create_matrix(float *x, float *b, float **a, int N) {
srand ( time ( NULL));
int i, j;
for (i = 0; i < N; i++) {
x[i] = 0.0;
b[i] = (float)rand()/(RAND_MAX*2.0-1.0);
a[i][i] = 2.0+(float)rand()/(RAND_MAX*2.0-1.0);
for (j = 0; j < i; j++) 
a[i][j] = (float)rand()/(RAND_MAX*2.0-1.0);;
} 
}
void calculate_x_matrix(float *x, float *b, float **a, int *flag, int N) {
int i, j;
float sum;
#pragma omp parallel for default(shared) private(i,j, sum)
for (i = 0; i < N; i++) {
sum = 0.0;
for (j = 0; j < i; j++) {
#pragma omp flush(flag)
while (!flag[j]) { 
#pragma omp flush (flag) 
}
sum = sum + (x[j] * a[i][j]);
}	
x[i] = (b[i] - sum) / a[i][i];
#pragma omp atomic write
flag[i] = 1;
#pragma omp flush (flag)
}
}
void print_results(float *x, int N) {
int i;
for (i = 0; i < N; i++) {
printf ("%f \n", x[i]);
}
}
void validate_results(float *x, float *b, float **a, int N) {
int i, j;
float sum;
for (i = 0; i < N; i++) {
sum = 0.0;
for (j = 0; j < N; j++) {
sum = sum + (x[j] * a[i][j]);
if (abs(b[i] - sum) > 0.00001) {
printf("%f != %f\n", sum, b[i]);
printf("Validation Failed...\n");
}
}
}
}
void display_time(double start, double end){
(void) printf("Time spent for sorting: %f seconds\n", (end-start));
}
