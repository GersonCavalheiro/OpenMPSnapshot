#include <stdio.h>
#include <omp.h>
int main()
{
int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
int n = 10;
int sum = 0;
double itime = omp_get_wtime();
printf("\nInitial Array Declaration:\n");
for (int i = 0; i < n; i++)
{
printf("%d ", arr[i]);
}
printf("\n");
#pragma omp parallel for schedule(dynamic)
#pragma omp reduction(+:sum)
for (int i = 0; i < n; i++)
{
printf("Thread %d - sum=%d\n", omp_get_thread_num(), sum);
sum += arr[i];
}
printf("\n\nArray Sum after construct: %d\n", sum);
double ftime = omp_get_wtime();
double timetaken = ftime - itime;
printf("\nTime Taken = %f", timetaken);
return 0;
}
