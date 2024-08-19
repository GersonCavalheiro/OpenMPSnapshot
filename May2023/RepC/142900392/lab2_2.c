#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#define CHUNKSIZE 4
int main (int argc, char *argv[]) 
{
double parallel_start, parallel_end,seq_start,seq_end,parallel_time,seq_time;  
srand(time(NULL));
printf("==========EXERCISE 2==========\n");
printf("    MATRIX ADDITION AND SUBTRACTION\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");
int i,j,r,c;
FILE *fptr;
printf("Enter the number of rows: ");
scanf("%d",&r);
printf("\nEnter the number of columns: ");
scanf("%d",&c);
float **a = (float **) malloc (r*sizeof(float *));
float **b = (float **) malloc (r*sizeof(float *));
float **sum = (float **) malloc (r*sizeof(float *));
float **diff = (float **) malloc (r*sizeof(float *));
float n;
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_2_A.txt", "w");
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
printf("\nMatrix A Generated\n");
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_2_B.txt", "w");
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
printf("\nMatrix B Generated\n\n");
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_2_A.txt", "r");
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
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_2_B.txt", "r");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < r; ++i)
{
b[i]=(float *) malloc (c*sizeof(float));
for (j = 0; j < c; ++j)
{
fscanf(fptr,"%f",&n);
b[i][j]=n;
}
}
fclose(fptr);
for (i=0; i<r; i++)
{
sum[i]=(float *) malloc (c*sizeof(float));
diff[i]=(float *) malloc (c*sizeof(float));
}
seq_start = omp_get_wtime();
for (i=0; i<r; i++)
{
for(j=0;j<c;j++)
{sum[i][j] = a[i][j] + b[i][j];
diff[i][j] = a[i][j] - b[i][j];}
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
for (i=0; i<r; i++)
{
for(j=0;j<c;j++)
{sum[i][j] = a[i][j] + b[i][j];
diff[i][j] = a[i][j] - b[i][j];
printf("Thread %d: sum[%d][%d]= %f\n",tid,i,j,sum[i][j]);}
}
}
parallel_end = omp_get_wtime();
parallel_time = parallel_end - parallel_start; 
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_2_sum.txt", "w");
for (i = 0; i < r; ++i)
{
for (j = 0; j < c; ++j)
{
fprintf(fptr,"%f ",sum[i][j]);
}
fprintf(fptr, "\n" );
}
fclose(fptr);
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_2_diff.txt", "w");
for (i = 0; i < r; ++i)
{
for (j = 0; j < c; ++j)
{
fprintf(fptr,"%f ",diff[i][j]);
}
fprintf(fptr, "\n" );
}
fclose(fptr);
printf("Parallel Time :%lf",parallel_time);
printf("\nSequential Time :%lf\n",seq_time);
return (0);
}
