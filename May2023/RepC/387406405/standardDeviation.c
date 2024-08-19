#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
float *create_rand_nums(int num_elements) 
{
float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
assert(rand_nums != NULL);
int i;
for (i = 0; i < num_elements; i++) 
{
rand_nums[i] = (rand() / (float)RAND_MAX);
}
return rand_nums;
}
int main(int argc, char** argv) 
{
if (argc != 2) 
{
fprintf(stderr, "Usage: avg num_elements_per_proc\n");
exit(1);
}
int num_elements_per_proc = atoi(argv[1]);
clock_t begin, end;
double time_spent;
MPI_Init(NULL, NULL);
int world_rank;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
int world_size;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
begin = clock();
srand(time(NULL)*world_rank); 
float *rand_nums = NULL;
rand_nums = create_rand_nums(num_elements_per_proc);
float local_sum = 0;
int i,n,m;
printf("enter the number of threads to be used to calculate local_sum:\n");
scanf("%d",&n);
printf("enter the number of threads to be used to calculate local_sq_diff:\n");
scanf("%d",&m);
#pragma omp parallel for reduction(+: local_sum) num_threads(n)
for (i = 0; i < num_elements_per_proc; i++) 
{
local_sum += rand_nums[i];
}
printf("Used %dthreads under process %d to calculate local_sum\n", n,world_rank);
float global_sum;
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM,MPI_COMM_WORLD);
float mean = global_sum / (num_elements_per_proc * world_size);
float local_sq_diff = 0;
#pragma omp parallel for reduction(+: local_sq_diff) num_threads(m)
for (i = 0; i < num_elements_per_proc; i++) 
{
printf("%d ", omp_get_thread_num());
local_sq_diff += (rand_nums[i] -mean) * (rand_nums[i] -mean);
}
printf("\nUsed %dthreads under process %d to calculate local_sq_diff\n", m,world_rank);
float global_sq_diff;
MPI_Reduce(&local_sq_diff, &global_sq_diff, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
end = clock();
if (world_rank == 0) 
{
float stddev = sqrt(global_sq_diff /(num_elements_per_proc * world_size));
printf("Mean -%f, Standard deviation = %f\n", mean, stddev);
time_spent = (double)(end -begin) / CLOCKS_PER_SEC;
printf("Time spent : %f\n\n", time_spent);
}
free(rand_nums);
MPI_Barrier(MPI_COMM_WORLD);
MPI_Finalize();
}
