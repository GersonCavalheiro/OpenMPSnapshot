#include <stdio.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#define OUTPUTS	1
#define INPUTS  2
#define SAMPLES 1
void Matrix_mult(float *A, float *B, float *res,
int M, int L, int N, int thread_count);
void Random_assign(float *Matrix, int rows, int cols);
void Activ_func(float *in,float *out,int row, int col, int thread_count);
void Usage();
int main(int argc, char *argv[])
{
if (argc < 2) Usage();
int NEURONS;
int thread_count;
double start, finish;
NEURONS 		 = strtol(argv[1], NULL, 10);
thread_count = strtol(argv[2], NULL, 10);
float X[INPUTS+1][SAMPLES] = {{0},{0}};
float Wx[NEURONS][INPUTS+1];
float IDF1[NEURONS][SAMPLES];
float ATV1[NEURONS][SAMPLES];
float Wy[OUTPUTS][NEURONS+1];
float IDF2[OUTPUTS][SAMPLES];
float ATV2[OUTPUTS][SAMPLES];
Random_assign(Wx[0],NEURONS,INPUTS+1);
Random_assign(Wy[0],OUTPUTS, NEURONS+1);
start = omp_get_wtime();
Matrix_mult(Wx[0],X[0],IDF1[0],NEURONS,INPUTS+1,SAMPLES, thread_count);
Activ_func(IDF1[0],ATV1[0],NEURONS,SAMPLES, thread_count);
Matrix_mult(Wy[0],ATV1[0],IDF2[0],OUTPUTS,NEURONS+1,SAMPLES, thread_count);
Activ_func(IDF2[0],ATV2[0],OUTPUTS,SAMPLES, thread_count);
finish = omp_get_wtime();
printf("For %d threads, the time taken was: %lf seconds \n", thread_count, finish-start);
#ifdef DEBUG
int i,j;
for(i = 0;i< OUTPUTS;i++)
{
for(j=0; j < SAMPLES;j++)
{
std::cout << std::setprecision(2) << std::fixed << ATV2[i][j] << "\t";
}
std::cout << "\n";
}
#endif
return 0;
}
void Usage()
{
std::cout << "Some arguments are missing." << std::endl;
std::cout << "Usage: ./feed <NEURONS> <thread_count>" << std::endl;
exit(0);
}
void Matrix_mult(float *A, float *B, float *res,
int M, int L, int N, int thread_count)
{
int i, j, k;
#pragma omp parallel num_threads(thread_count) private(i, j, k) shared(A,B, res,M,L,N)
{
double result = 0;
#pragma omp for schedule(static) 
for (i = 0; i < M; i++){
for (j = 0; j < N; j++) {
result = 0;
for (k = 0; k < L; k++) {
result += (*(A + i*L + k))*(*(B + k*N + j));
}
*(res+ j +i*N) = (float)result; 
}
}
}
}
void Activ_func(float *in,float *out,int rows, int cols,int thread_count)
{ 
int i,j;
#pragma omp parallel num_threads(thread_count) private (i,j) shared(in, out, rows, cols)
{
double result=0;
#pragma omp for schedule(static)
for(i=0;i < rows;i++)
{
for(j = 0;j< cols;j++)
{
result = 1.0/(1.0 + exp(-(*(in + i*cols+j))));
}
*(out + i*cols+j) = (float)result; 
}
}
} 
void Random_assign(float *Matrix, int rows, int cols)
{
srand(time(NULL)); 
int i,j;
for(i=0; i<rows; i++)
{
for(j=0; j<cols; j++)
{
*Matrix++ = (float)rand()/(float)RAND_MAX;
}
}
}
