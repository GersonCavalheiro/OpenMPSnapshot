#include<stdlib.h> 
#include<stdio.h> 
#include<omp.h>
#include<time.h>
int partition (int *arr, int low, int high) 
{ 
int temp;
int pivot = arr[high];    
int i = (low - 1);  
int j;
for (j = low; j <= high- 1; j++) 
{ 
if (arr[j] <= pivot) 
{ 
i++; 
temp = arr[i];
arr[i ] = arr[j];
arr[j] = temp;
} 
}
temp = arr[i + 1];
arr[i + 1] = arr[high];
arr[high] = temp; 
return (i + 1); 
} 
void quickSort(int *arr, int low, int high) 
{ 
if (low < high) 
{ 
int i;        
int pi = partition(arr, low, high);
int *arr_copy = (int*) malloc(sizeof(arr));
for(i=0;i<sizeof(arr)/sizeof(int);i++)
arr_copy[i]=arr[i];
double seq_start = omp_get_wtime();
quickSort(arr_copy, low, pi - 1);
quickSort(arr_copy, pi + 1, high);
printf("Sequential Time : %lf\n",omp_get_wtime()-seq_start);
double parallel_start = omp_get_wtime();
#pragma omp parallel
{	
#pragma omp sections
{
#pragma omp section
{printf("Thread ID: %d\n",omp_get_thread_num()); 
quickSort(arr, low, pi - 1); }
#pragma omp section
{printf("Thread ID: %d\n",omp_get_thread_num()); 
quickSort(arr, pi + 1, high); }
}
}
printf("Parallel Time: %lf\n",omp_get_wtime()-parallel_start);
} 
} 
int main() 
{ 
srand(time(NULL));
printf("==========EXERCISE 3a==========\n");
printf("    QUICKSORT\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");    
int i,n,m;
printf("Enter array size: "); 
scanf("%d",&n);
int *arr = (int*) malloc(sizeof(int)*n);
FILE *fptr;
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_3a_input.txt", "w");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < n; ++i)
{
m = rand()%1000;
fprintf(fptr,"%d ", m);
}
fclose(fptr);
printf("\nArray Generated\n");
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_3a_input.txt", "r");
if(fptr == NULL)
{
printf("Error!");
exit(1);
}
for (i = 0; i < n; ++i)
{
fscanf(fptr,"%d",&m);
arr[i]=m;
}
fclose(fptr);
quickSort(arr, 0, n-1);
printf("Sorted array: "); 
for (i=0; i < n; i++) 
printf("%d ", arr[i]);
printf("\n");
fptr = fopen("/home/dhanya/PDC_Lab/Assignment2/lab2_3a_output.txt", "w");
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
