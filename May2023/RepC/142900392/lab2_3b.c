#include<stdlib.h> 
#include<stdio.h> 
#include<omp.h>
#include<time.h>
void merge(int *arr, int lower, int mid, int upper) 
{ 
int i, j, k; 
int n1 = mid - lower + 1; 
int n2 =  upper - mid; 
int *L = (int*) malloc(sizeof(int)*n1);
int *R = (int*) malloc(sizeof(int)*n2);
for (i = 0; i < n1; i++) 
L[i] = arr[lower + i]; 
for (j = 0; j < n2; j++) 
R[j] = arr[mid + 1+ j]; 
i = 0;  
j = 0; 
k = lower; 
while (i < n1 && j < n2) 
{ 
if (L[i] <= R[j]) 
{ 
arr[k] = L[i]; 
i++; 
} 
else
{ 
arr[k] = R[j]; 
j++; 
} 
k++; 
} 
while (i < n1) 
{ 
arr[k] = L[i]; 
i++; 
k++; 
} 
while (j < n2) 
{ 
arr[k] = R[j]; 
j++; 
k++; 
} 
} 
void mergeSort(int *arr, int lower, int upper) 
{ 
if (lower < upper) 
{ 
int i;        
int mid = lower+(upper-lower)/2; 
int *arr_copy = (int*) malloc(sizeof(arr));
for(i=0;i<sizeof(arr)/sizeof(int);i++)
arr_copy[i]=arr[i];
double seq_start = omp_get_wtime();
mergeSort(arr_copy, lower, mid);
mergeSort(arr_copy, mid+1, upper);
printf("Sequential Time : %lf\n",omp_get_wtime()-seq_start);
double parallel_start = omp_get_wtime();
#pragma omp parallel 
{	
#pragma omp sections
{
#pragma omp section
{printf("Thread ID: %d\n",omp_get_thread_num());			
mergeSort(arr, lower, mid); 
}
#pragma omp section
{printf("Thread ID: %d\n",omp_get_thread_num());
mergeSort(arr, mid+1, upper); 
}
}
}
printf("Parallel Time: %lf\n",omp_get_wtime()-parallel_start);
merge(arr, lower, mid, upper); 
} 
} 
int main() 
{ 
srand(time(NULL));
printf("==========EXERCISE 3b==========\n");
printf("    MERGESORT\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");    
int i,n,m;
printf("Enter array size: "); 
scanf("%d",&n);
int *arr = (int*) malloc(sizeof(int)*n);
FILE *fptr;
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_3b_input.txt", "w");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < n; ++i)
{
m = rand()%1000;arr[i]=m;
fprintf(fptr,"%d ", m);
}
fclose(fptr);
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_3b_input.txt", "r");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < n; ++i)
{
fscanf(fptr,"%d ", &m);
arr[i]=m;
}
fclose(fptr);
mergeSort(arr, 0, n - 1); 
printf("Sorted array: "); 
for (i=0; i < n; i++) 
printf("%d ", arr[i]); 
printf("\n");
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_3b_output.txt", "w");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < n; ++i)
{
fprintf(fptr,"%d ", arr[i]);
}
fclose(fptr);
return 0; 
} 
