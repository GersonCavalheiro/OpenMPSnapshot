#ifndef OMP_BUCKET_SORT
#define OMP_BUCKET_SORT
#include <omp.h>
#include "../sorting/insertion_sort.h"
#include "../utils/utils_sort.h"
#include "../utils/utils_sort.h"
#ifndef OMP_THREADS
#define OMP_THREADS 2
#endif
int omp_isqrt(int value)
{
int res = 0;
while ((res + 1) * (res + 1) <= value)
{
res += 1;
}
return res;
}
void omp_bucket_sort(int arr[], int n)
{
int bucket_count = (omp_isqrt(n));
int bucket_interval = n / bucket_count;
while (bucket_count * bucket_interval < n)
{
bucket_interval += 1;
}
int buckets[bucket_count][bucket_interval*2]; 
int bucket_index_count[bucket_count];
#pragma omp parallel for schedule(static) 
for (int i = 0; i < bucket_count; i++){
bucket_index_count[i] = 0;
}
int max = arr[0];
#pragma omp parallel for schedule(static) 
for (int i = 1; i < n; i++)
{
if (arr[i] > max){
max = arr[i];
}
}
max++;
for (int i = 0; i < n; i++)
{
int bucket_index = (bucket_count * arr[i]) / (max);
buckets[bucket_index][bucket_index_count[bucket_index]] = arr[i];
bucket_index_count[bucket_index]++;
}
#pragma omp parallel for schedule(static) 
for (int i = 0; i < bucket_count; i++)
{
insertion_sort(buckets[i], bucket_index_count[i]);
}
int arr_index = 0;
for (int i = 0; i < bucket_count; i++)
{
for (int j = 0; j < bucket_index_count[i]; j++) 
{
arr[arr_index] = buckets[i][j];
arr_index++;
}
}
}
#endif