#include <omp.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
int printMatrix(int rows, int cols, float *a);
int readMatrix(unsigned int rows, unsigned int cols, float *a, const char *filename);
int writeMatrix(unsigned int rows, unsigned int cols, float *a, const char *filename);
int y;
int w;
int v;
int main(int argc, char *argv[])
{
y = atoi(argv[1]);
w = atoi(argv[2]);
v = atoi(argv[3]);
char arqA[10];
char arqB[10];
char arqC[10];
char arqD[10];
strcpy(arqA, argv[4]);
strcpy(arqB, argv[5]);
strcpy(arqC, argv[6]);
strcpy(arqD, argv[7]);
float *matrizA = (float *)malloc(y * w * sizeof(float));
float *matrizB = (float *)malloc(w * v * sizeof(float));
float *matrizC = (float *)malloc(v * 1 * sizeof(float));
float *matrizD = (float *)malloc(y * 1 * sizeof(float));
float *aux = (float *)malloc(y * v * sizeof(float));
readMatrix(y, w, matrizA, arqA);
readMatrix(w, v, matrizB, arqB);
readMatrix(v, 1, matrizC, arqC);
int i;
int j;
int k;
double soma = 0.0;
double time_spent = 0.0;
struct timeval start, end;
gettimeofday(&start, NULL);
#pragma omp parallel shared(matrizA, matrizB, matrizC, aux, y, v) private(i, j, k)
{
#pragma omp for
for (i = 0; i < y; i++)
{
for (j = 0; j < v; j++)
{
aux[i * v + j] = 0.0;
for (k = 0; k < w; k++)
{
aux[i * v + j] = aux[i * v + j] + matrizA[i * w + k] * matrizB[k * v + j];
}
}
}
#pragma omp for
for (i = 0; i < y; i++)
{
for (j = 0; j < 1; j++)
{
matrizD[i * 1 + j] = 0.0;
for (k = 0; k < v; k++)
{
matrizD[i * 1 + j] = matrizD[i * 1 + j] + aux[i * v + k] * matrizC[k * 1 + j];
}
}
}
}
#pragma omp parallel for shared(matrizD) private(i,j) reduction(+:soma) num_threads(4)
for (i = 0; i < y; i++)
{
for (j = 0; j < 1; j++)
{
soma += matrizD[i * 1 + j];
}
}
gettimeofday(&end, NULL);
long seconds = (end.tv_sec - start.tv_sec);
long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
writeMatrix(y, 1, matrizD, arqD);
printf("%.2f\n", soma);
}
int readMatrix(unsigned int rows, unsigned int cols, float *a, const char *filename)
{
FILE *pf;
pf = fopen(filename, "r");
if (pf == NULL)
return 0;
register unsigned int i, j;
char k[15];
for (i = 0; i < rows; ++i)
{
for (j = 0; j < cols; ++j)
{
fscanf(pf, "%s", k);
a[i * cols + j] = strtof(k, NULL);
}
}
fclose(pf);
return 1;
}
int printMatrix(int rows, int cols, float *a)
{
register unsigned int i, j;
for (i = 0; i < rows; i++)
{
for (j = 0; j < cols; j++)
{
printf("%.2f\t", a[i * cols + j]);
}
printf("\n");
}
return 1;
}
int writeMatrix(unsigned int rows, unsigned int cols, float *a, const char *filename)
{
FILE *pf;
pf = fopen(filename, "w");
if (pf == NULL)
return 0;
register unsigned int i, j;
for (i = 0; i < rows; ++i)
{
for (j = 0; j < cols; ++j)
{
fprintf(pf, "%.2f\n", a[i * cols + j]);
}
}
fclose(pf);
return 1;
}