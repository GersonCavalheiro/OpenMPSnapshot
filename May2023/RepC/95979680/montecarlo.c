#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../timer.h"
const int MAX_THREADS = 1024;
void Get_args(int argc, char* argv[], long long *thread_count, long long *total_tosses);
void Usage(char* prog_name);
int main(int argc, char* argv[]) 
{
int i, seed; 
double x, y, estimate_pi;
double start, finish, elapsed;
long long thread_count, total_tosses, total_in_circle;
struct drand48_data drand_buf;                           
Get_args(argc, argv, &thread_count, &total_tosses);
omp_set_num_threads(thread_count);
total_in_circle = 0;
GET_TIME(start);
#pragma omp parallel private(i,x,y,seed,drand_buf) shared(total_tosses)
{
seed = omp_get_thread_num();
srand48_r(seed,&drand_buf);
#pragma omp for reduction(+: total_in_circle)
for (i = 0; i < total_tosses; i++)
{
drand48_r(&drand_buf,&x);
drand48_r(&drand_buf,&y);
if (x*x + y*y <= 1) total_in_circle++;
}
}
GET_TIME(finish);
elapsed = finish - start;
estimate_pi = (double)(4.0*total_in_circle) / (double)total_tosses;
printf("Total in circle = %lld\n",total_in_circle);
printf("With number_of_tosses = %lld,\n", total_tosses);
printf("   pi = %.15lf\n",M_PI);
printf("   Our estimate of pi = %.15f\n", estimate_pi);
printf("The elapsed time is %.10lf seconds\n", elapsed);
return 0;
}  
void Get_args(int argc, char* argv[], long long *thread_count, long long *total_tosses) 
{
if (argc != 3) Usage(argv[0]);
*thread_count = strtoll(argv[1], NULL, 10);  
if (*thread_count <= 0 || *thread_count > MAX_THREADS) Usage(argv[0]);
*total_tosses = strtoll(argv[2], NULL, 10);
if (*total_tosses <= 0) Usage(argv[0]);
}  
void Usage(char* prog_name) 
{
fprintf(stderr, "usage: %s <number of threads> <number of tosses>\n", prog_name);
fprintf(stderr, "   n is the number of tosses to the dartboard\n");
fprintf(stderr, "   n should be evenly divisible by the number of threads\n");
exit(0);
}  
