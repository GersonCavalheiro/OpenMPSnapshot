#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <omp.h>
float *create_rand_nums(int num_elements) 
{
float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
assert(rand_nums != NULL);
int i,n;
printf("enter the number of threads to be used to create array of random numbers:\n");
scanf("%d",&n);
#pragma omp parallel for num_threads(n)
for (i = 0; i < num_elements; i++) 
{
rand_nums[i] = (rand() / (float)RAND_MAX);
}
return rand_nums;
}
float compute_avg(float *array, int num_elements) 
{
float sum = 0.f;
int i;
#pragma omp parallel for reduction(+: sum) num_threads(10)
for (i = 0; i < num_elements; i++) 
{
sum += array[i];
}
return sum / num_elements;
}
int main(int argc, char** argv) 
{
if (argc != 2) 
{
fprintf(stderr, "Usage: avg num_elements_per_proc\n");
exit(1);
}
int num_elements_per_proc = atoi(argv[1]);
srand(time(NULL));
clock_t begin, end;
double time_spent;
MPI_Init(NULL, NULL);
int world_rank;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
int world_size;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
begin = clock();
float *rand_nums = NULL;
if (world_rank == 0) 
{
rand_nums = create_rand_nums(num_elements_per_proc * world_size);
}
float *sub_rand_nums = (float *)malloc(sizeof(float) * num_elements_per_proc);
assert(sub_rand_nums != NULL);
MPI_Scatter(rand_nums, num_elements_per_proc, MPI_FLOAT, sub_rand_nums,num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
float sub_avg = compute_avg(sub_rand_nums, num_elements_per_proc);
float *sub_avgs = (float *)malloc(sizeof(float) * world_size);
assert(sub_avgs != NULL);
MPI_Allgather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, MPI_COMM_WORLD);
end = clock();
float avg = compute_avg(sub_avgs, world_size);
if (world_rank == 0) 
{
printf("Avg of all elements from proc %d is %f\n", 0, avg);
time_spent = (double)(end -begin) / CLOCKS_PER_SEC;
printf("Time spent : %f\n\n", time_spent);
free(rand_nums);
}
free(sub_avgs);
free(sub_rand_nums);
MPI_Barrier(MPI_COMM_WORLD);
MPI_Finalize();
}
