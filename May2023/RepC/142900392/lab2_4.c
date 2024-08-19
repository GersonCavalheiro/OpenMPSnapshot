#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#define CHUNKSIZE 4
int main (int argc, char *argv[]) 
{
double parallel_start, parallel_end,seq_start,seq_end,parallel_time,seq_time;  
srand(time(NULL));
printf("==========EXERCISE 4==========\n");
printf("    MATRIX TRANSPOSE\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");
int i,j,r,c;
FILE *fptr;
printf("Enter the number of rows: ");
scanf("%d",&r);
printf("\nEnter the number of columns: ");
scanf("%d",&c);
float **a = (float **) malloc (r*sizeof(float *));
float **b = (float **) malloc (c*sizeof(float *));
float n;
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_4_input.txt", "w");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < r; ++i)
{
for (j = 0; j < c; ++j)
{
n = rand()%1000;
fprintf(fptr,"%f ", n);
}
fprintf(fptr,"\n");
}
fclose(fptr);
printf("\nMatrix Generated\n");
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_4_input.txt", "r");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < r; ++i)
{
a[i]=(float *) malloc (c*sizeof(float));
for (j = 0; j < c; ++j)
{
fscanf(fptr,"%f",&n);
a[i][j]=n;
}
}
fclose(fptr);
seq_start = omp_get_wtime();
for (i=0; i<c; i++)
{
b[i]=(float *) malloc (r*sizeof(float));
for(j=0;j<r;j++)
{b[i][j] = a[j][i];}
}
seq_end = omp_get_wtime();
seq_time = seq_end - seq_start; 
int nthreads, tid,chunk;
chunk = CHUNKSIZE;
parallel_start = omp_get_wtime();
#pragma omp parallel shared(a,b,nthreads,chunk) private(i,j,tid)
{
tid = omp_get_thread_num();
if (tid == 0)
{
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}
printf("Thread %d starting...\n",tid);
#pragma omp for schedule(guided,chunk)
for (i=0; i<c; i++)
{
for(j=0;j<r;j++)
{b[i][j] = a[j][i];
printf("Thread %d: b[%d][%d]= %f\n",tid,i,j,b[i][j]);}
}
}
parallel_end = omp_get_wtime();
parallel_time = parallel_end - parallel_start; 
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_4_output.txt", "w");
for (i = 0; i < c; ++i)
{
for (j = 0; j < r; ++j)
{
fprintf(fptr,"%f ",b[i][j]);
}
fprintf(fptr, "\n" );
}
fclose(fptr);
printf("Parallel Time :%lf",parallel_time);
printf("\nSequential Time :%lf\n",seq_time);
return (0);
}
