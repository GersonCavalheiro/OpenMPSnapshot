#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
/
#define N 100000            
#define Nv 1000             
#define Nc 100              
#define THR_KMEANS 0.000001 
void createVectors();
void initCenters();
void classification();
void estimateCenters(void);
int terminate(void);
void print2dFloatMatrix(float *mat, int x, int y);
void print1dIntMatrix(int *mat, int x);
float vectors[N][Nv];
float centers[Nc][Nv];
int classes[N]; 
float prevDistSum;
float currentDistSum;
float dist;
float distmin;
int indexmin;
int vecsInCenter;
float newCenterValues[Nv];
int main()
{
createVectors();
initCenters();
do
{
classification();
estimateCenters();
} while (terminate());
return 0;
}
void createVectors()
{
for (int i = 0; i < N; i++)
for (int j = 0; j < Nv; j++)
vectors[i][j] = (float)rand() / (float)RAND_MAX;
}
void initCenters()
{
memcpy(centers, vectors, sizeof(centers));
}
void classification()
{
int i, j, k;
prevDistSum = currentDistSum;   
currentDistSum = 0;             
#pragma omp parallel for private(i, j, k, dist, distmin, indexmin) reduction(+:currentDistSum) schedule(static)
for (i = 0; i < N; i++)
{
distmin = 0;        
indexmin = 0;               
for (k = 0; k < Nv; k++)    
{
distmin += (vectors[i][k] - centers[0][k]) * (vectors[i][k] - centers[0][k]);
}
for (j = 1; j < Nc; j++)
{
dist = 0;
#pragma omp simd
for (int k = 0; k < Nv; k++)
dist += (vectors[i][k] - centers[j][k]) * (vectors[i][k] - centers[j][k]);
if (dist < distmin)
{
distmin = dist; 
indexmin = j;
}
}
classes[i] = indexmin; 
currentDistSum += distmin;
}
}
void estimateCenters(void)
{
for (int i = 0; i < Nc; i++) 
{
vecsInCenter = 0;
memset(newCenterValues, 0, sizeof(newCenterValues));
for (int j = 0; j < N; j++) 
if (classes[j] == i) 
{
vecsInCenter++;
for (int k = 0; k < Nv; k++)
newCenterValues[k] += vectors[j][k];
}
for (int k = 0; k < Nv; k++)
centers[i][k] = newCenterValues[k] / vecsInCenter;
}
}
int x = 0;
int terminate(void)
{
printf("%f\n", currentDistSum);
if (x == 10)
return 0;
x++;
return 1;
}
void print2dFloatMatrix(float *mat, int x, int y)
{
for (int i = 0; i < x; i++)
{
printf("\n%d\n", i);
for (int j = 0; j < y; j++)
printf("%f ", *(mat + (y * i) + j));
}
printf("\n");
}
void print1dIntMatrix(int *mat, int x)
{
for (int x = 0; x < N; x++)
printf("%d\n", mat[x]);
}