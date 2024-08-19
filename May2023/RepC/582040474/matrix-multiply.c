#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
short int MAT_DIMS = 1000;
short int NO_THREADS = 12;
void initMat(short int *matA, short int *matB)
{
for (short int i = 0; i < MAT_DIMS; i++)
{
for (short int j = 0; j < MAT_DIMS; j++)
{
matA[i * MAT_DIMS + j] = rand() % 5;
matB[i * MAT_DIMS + j] = rand() % 5;
}
}
}
void matMultiplySequential(short int *matA, short int *matB, unsigned short int *mat)
{
for (short int i = 0; i < MAT_DIMS; i++)
{
for (short int j = 0; j < MAT_DIMS; j++)
{
for (short int k = 0; k < MAT_DIMS; k++)
{
mat[i * MAT_DIMS + j] += matA[i * MAT_DIMS + k] * matB[k * MAT_DIMS + j];
}
}
}
}
void matMultiplyOmp(short int *matA, short int *matB, unsigned short int *mat)
{
#pragma omp parallel num_threads(NO_THREADS) shared(matA, matB, mat)
{
short int numThreads = omp_get_num_threads();
short int threadNo = omp_get_thread_num();
unsigned short int temp[MAT_DIMS];
for (short int i = threadNo; i < MAT_DIMS; i += numThreads)
{
memset(temp, 0, MAT_DIMS * sizeof(short int));
for (short int j = 0; j < MAT_DIMS; j++)
{
for (short int k = 0; k < MAT_DIMS; k++)
{
temp[k] += matA[i * MAT_DIMS + j] * matB[j * MAT_DIMS + k];
}
}
memcpy(&mat[i * MAT_DIMS], temp, MAT_DIMS * sizeof(short int));
}
}
}
void printMat(unsigned short int *mat)
{
for (short int i = 0; i < MAT_DIMS; i++)
{
for (short int j = 0; j < MAT_DIMS; j++)
{
printf("%d ", mat[i * MAT_DIMS + j]);
}
printf("\n");
}
printf("\n");
}
int main()
{
srand(time(NULL));
short int *matA;
short int *matB;
unsigned short int *resultMat;
double startTime;
for (; MAT_DIMS > 0; MAT_DIMS -= 100)
{
matA = (short int *)malloc(MAT_DIMS * MAT_DIMS * sizeof(short int));
matB = (short int *)malloc(MAT_DIMS * MAT_DIMS * sizeof(short int));
initMat(matA, matB);
startTime = omp_get_wtime();
for (short int t = 0; t < 5; t++)
{
resultMat = (short int *)calloc(MAT_DIMS * MAT_DIMS, sizeof(short int));
matMultiplyOmp(matA, matB, resultMat);
}
printf("No. of Threads %d, Dims %d, Time elapsed: %f\n", NO_THREADS, MAT_DIMS, (omp_get_wtime() - startTime) / 5);
}
return 0;
}
