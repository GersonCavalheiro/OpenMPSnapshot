#include <cstdlib>
#include <cstdio>


#pragma omp declare target
float offload_allocate(int xdim) {
float arr[xdim];
float sum = 0.0;

for (int i = 0; i < xdim; i++)
arr[i] = i;

for (int i = 0; i < xdim; i++)
sum += arr[i];

return sum;
}
#pragma omp end declare target

void test_offload_allocate(int N) {
float sum = 0.0;
#pragma omp target data map(tofrom:sum) map(to:N)
{
sum = offload_allocate(N);
}
printf("Sum %f \n", sum);
}

#pragma omp declare target
float offload_allocate_malloc(int xdim) {
float *arr = (float*) malloc((sizeof(float)*xdim));
float sum = 0.0;

for (int i = 0; i < xdim; i++)
arr[i] = i;

for (int i = 0; i < xdim; i++)
sum += arr[i];

printf("Sum in device %f\n", sum);

free(arr);

return sum;
}
#pragma omp end declare target

void test_offload_allocate_malloc(int N) {
float sum = 0.0;
#pragma omp target map(from:sum) map(to:N)
{
sum = offload_allocate_malloc(N);

}
printf("Sum %f \n", sum);
}
