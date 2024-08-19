#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
int main()
{
int n,i,sum;
double start_time,end_time;
printf("==========EXERCISE 10==========\n");
printf("    SUM OF ARRAY ELEMENTS\n\t\tS. DHANYA ABHIRAMI\n\t\t16BCE0965\n");	
printf("Enter size of array: ");
scanf("%d",&n);
int *a;
a=(int *) malloc(sizeof(int)*n);
printf("Enter elements of the array: ");
for (i=0;i<n;i++)
{
scanf("%d",&a[i]);
}
printf("Parallel Execution without reduce clause\n");
sum=0;
start_time = omp_get_wtime();
#pragma omp parallel for private(x)
for (i=0;i<n;i++)
{
sum+=a[i];
}
end_time = omp_get_wtime();
printf("Execution Time without reduce clause: %lf\n", end_time-start_time);
printf("Sum = %d\n",sum);
printf("Parallel Execution\n");
sum=0;
start_time = omp_get_wtime();
#pragma omp parallel for reduction (+:sum)
for (i=0;i<n;i++)
{
sum+=a[i];
}
end_time = omp_get_wtime();
printf("Execution Time with reduce clause: %lf\n", end_time-start_time);
printf("Sum = %d\n",sum);
return 0;
}
