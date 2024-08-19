#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
void Init_matrix(int** mat, int n);
void Print_matrix(int** mat, int n);
void Floyd(int** mat, int n);
int main(void) {
int n;
struct timeval start_time, stop_time, elapsed_time;  
double  numFlops;
float gflops;
printf("How many vertices?\n");
scanf("%d", &n);
int* arr[n];
for (int i=0; i<n; i++)
arr[i] = (int *)malloc(n * sizeof(int));
Init_matrix(arr, n);   
gettimeofday(&start_time,NULL);
Floyd(arr, n);
gettimeofday(&stop_time,NULL);
timersub(&stop_time, &start_time, &elapsed_time); 
printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
numFlops = 2.0f*n*n*n/1000000000.0f;
gflops = numFlops/(elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
printf("GFlops :  %f .\n",gflops);
return 0;
}  
void Init_matrix(int ** mat, int n) {
int i, j,val;
int INFINTY=n-1;
for (i = 0; i < n; i++) {
for (j = 0; j < n; j++)
if (i == j)
mat[i][j]=0;
else {
if ((i==j+1)|| (j==i+1)||((i==0)&&(j==n-1))||((i==n-1)&&(j==0)))
mat[i][j]=1;
else
mat[i][j]= n;
}
}
}  
void Print_matrix(int**  mat, int n) {
int i, j;
for (i = 0; i < n; i++) {
for (j = 0; j < n; j++)
printf("%d ", mat[i][j]);
printf("\n");
}
}  
void Floyd(int ** restrict  arr, int n) {
int k, i, j;
#pragma acc data copy(arr[0:n][0:n])
{
for (k = 0; k < n; k++) {
#pragma acc region
{	 
#pragma acc loop independent gang vector(8)
for (i = 0; i < n; i++)
#pragma acc loop independent gang vector(256)
for (j = 0; j < n; j++) {
arr[i][j] =fmin(arr[i][j], arr[i][k] + arr[k][j]);
}
}
}
}  
} 
