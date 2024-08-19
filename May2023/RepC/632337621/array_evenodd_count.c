#include <stdio.h>
#include <omp.h>
int main()
{
int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int n = 10;
int e = 0, o = 0;
double itime = omp_get_wtime();
printf("\nInitial Array Declaration:\n");
for (int i = 0; i < n; i++)
{
printf("%d ", arr[i]);
}
#pragma omp parallel for
#pragma omp reduction(+:e) reduction(+:o)
for (int i = 0; i < n; i++)
{
printf("Thread %d - e = %d, o = %d\n", omp_get_thread_num(), e, o);
if (arr[i] % 2 == 0)
{
e++;
}
else
{
o++;
}
}
printf("\n\nEven numbers in array: %d", e);
printf("\nOdd numbers in array: %d\n", o);
double ftime = omp_get_wtime();
double timetaken = ftime - itime;
printf("\nTime Taken = %f", timetaken);
return 0;
}
