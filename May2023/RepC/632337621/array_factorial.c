#include <stdio.h>
#include <math.h>
#include <omp.h>
int main()
{
int arr[] = {2, 4, 5, 6, 10, 2, 3, 8, 7, 9};
int n = 10;
double itime = omp_get_wtime();
printf("\nArray of number:\n");
for (int i = 0; i < n; i++)
{
printf("%d ", arr[i]);
}
#pragma omp parallel for
for (int i = 0; i < n; i++)
{
int num = arr[i];
int fact = 1;
for (int j = 1; j <= num; j++)
{
fact *= j;
}
printf("\nFactorial of %d is %d from thread-%d", num, fact, omp_get_thread_num());
}
double ftime = omp_get_wtime();
double timetaken = ftime - itime;
printf("\n\nTime Taken = %f", timetaken);
return 0;
}
