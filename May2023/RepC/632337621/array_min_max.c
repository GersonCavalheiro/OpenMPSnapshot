#include <stdio.h>
#include <math.h>
#include <omp.h>
int main()
{
int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
int min_ = 9999, max_ = -9999;
double itime = omp_get_wtime();
printf("\nInitial Array Declaration:\n");
for (int i = 0; i < 9; i++)
{
printf("%d ", arr[i]);
}
#pragma omp parallel for schedule(dynamic)
#pragma omp firstprivate(min_) firstprivdate(max_)
#pragma omp reduction(min:min_) reduction(max:max_)
for (int i = 0; i < 9; i++)
{
printf("Thread %d starting...min=%d, max=%d\n", omp_get_thread_num(), min_, max_);
if (arr[i] > max_)
max_ = arr[i];
if ((arr[i]) < min_)
min_ = arr[i];
}
printf("\n\nMinimum in array: %d", min_);
printf("\nMaximum in array: %d\n", max_);
double ftime = omp_get_wtime();
double timetaken = ftime - itime;
printf("\nTime Taken = %f", timetaken);
return 0;
}
