#include <stdio.h>
#include <omp.h>
int main()
{
int arr[] = {1, 2, 3, 4, 5};
int n = 5;
printf("\nPrinting array elements using ordered construct\n");
#pragma omp parallel for ordered schedule(dynamic)
for (int i = 0; i < n; i++)
{
#pragma omp ordered
printf("%d ", arr[i]);
}
return 0;
}
