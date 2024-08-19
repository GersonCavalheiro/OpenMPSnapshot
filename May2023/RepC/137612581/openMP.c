#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define n 1000 
void showDistances(int** dist);
int main(int argc, char** argv) 
{
int i, j, k;
int** dist; 
dist = (int**)malloc(n*sizeof(int*));
for(i=0; i<n; i++)
dist[i] = (int*)malloc(n*sizeof(int));
time_t start, end;
time(&start);
srand(42);
for(i=0; i<n; i++)
for(j=0; j<n; j++)
if(i==j)
dist[i][j] = 0;
else
dist[i][j] = rand()%100;
showDistances(dist);	
time(&start);
#pragma omp parallel for private(i,j,k) shared(dist)
for(k=0; k<n; k++) 
for(i=0; i<n; i++)
for(j=0; j<n; j++)
if ((dist[i][k] * dist[k][j] != 0) && (i != j))
if(dist[i][j] > dist[i][k] + dist[k][j] || dist[i][j] == 0)
dist[i][j] = dist[i][k] + dist[k][j];
time(&end);
showDistances(dist);
printf("Total Elapsed Time %f sec\n", difftime(end, start));	
free(dist);
return 0;
}
void showDistances(int** dist) 
{
int i, j;
printf("     ");
for(i=0; i<n; ++i)
printf("N%d   ", i);
printf("\n");
for(i=0; i<n; ++i) {
printf("N%d", i);
for(j=0; j<n; ++j)
printf("%5d", dist[i][j]);
printf("\n");
}
printf("\n");
}	
