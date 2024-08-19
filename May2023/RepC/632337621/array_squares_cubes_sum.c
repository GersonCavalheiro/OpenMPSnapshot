#include <stdio.h>
#include <math.h>
#include <omp.h>
int main()
{
int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
int ssquares = 0, scubes = 0;
double itime = omp_get_wtime();
printf("\nInitial Array Declaration:\n");
for (int i = 0; i < 9; i++)
{
printf("%d ", arr[i]);
}
#pragma omp parallel for schedule(dynamic)
#pragma omp reduction(+:ssquares, +:scubes)
for (int i = 0; i < 9; i++)
{
printf("Thread %d - ssquares=%d, scubes=%d\n", omp_get_thread_num(), ssquares, scubes);
ssquares += pow(arr[i], 2);
scubes += pow(arr[i], 3);
}
printf("\n\nSum of Squares in array: %d", ssquares);
printf("\nSum of Cubes in array: %d\n", scubes);
double ftime = omp_get_wtime();
double timetaken = ftime - itime;
printf("\nTime Taken = %f", timetaken);
return 0;
}
