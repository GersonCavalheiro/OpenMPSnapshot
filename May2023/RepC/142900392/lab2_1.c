#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>
int main () 
{
printf("==============================EXERCISE 1==============================\n");
printf("    1D ARRAY ADDITION WITH ENVIRONMENTAL CLAUSES VARIABLE SCOPE\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");
int i,n;
float m;
double parallel_start, parallel_end,seq_start,seq_end,parallel_time,seq_time;  
FILE *fptr;
printf("Enter size of array: ");
scanf("%d",&n);
float *a=(float *) malloc (n*sizeof(float));
float *b=(float *) malloc (n*sizeof(float));
float *sum=(float *) malloc (n*sizeof(float));
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_1_A.txt", "w");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < n; ++i)
{
m = rand()%1000;
fprintf(fptr,"%f ", m);
}
fclose(fptr);
printf("\nArray A Generated\n");
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_1_B.txt", "w");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < n; ++i)
{
m = rand()%1000;
fprintf(fptr,"%f ", m);
}
fclose(fptr);
printf("\nArray B Generated\n\n");
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_1_A.txt", "r");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < n; ++i)
{
fscanf(fptr,"%f",&m);
a[i]=m;
}
fclose(fptr);
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_1_B.txt", "r");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < n; ++i)
{
fscanf(fptr,"%f",&m);
b[i]=m;
}
fclose(fptr);
seq_start = omp_get_wtime();
for (i=0; i<n; i++)
{
sum[i] = a[i] + b[i];
}
seq_end = omp_get_wtime();
seq_time = seq_end - seq_start; 
float temp=10.0;
printf("temp initialised = %f\n",temp );
parallel_start = omp_get_wtime();
#pragma omp parallel for lastprivate(temp)
for (i=0; i<n; i++)
{
printf("\nThread %d: \n",omp_get_thread_num());
printf("temp value = %f\n",temp );
temp = a[i] + b[i];
printf("temp updated = %f\n",temp );
sum[i] = temp;
printf("sum[%d]= %f\n",i,sum[i]);
}
parallel_end = omp_get_wtime();
parallel_time = parallel_end - parallel_start; 
printf("\ntemp outside = %f\n\n",temp );
printf("Parallel Time :%lf",parallel_time);
printf("\nSequential Time :%lf\n",seq_time);
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_1_sum.txt", "w");
for (i = 0; i < n; ++i)
{
fprintf(fptr,"%f ",sum[i]);
}
fclose(fptr); 
return (0);
}