#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#define MAX_THREAD_NUM 100
void print_help(char *executable);
int main(int argc, char *argv[])
{
char *executable = argv[0];
if (argc > 1 && strcmp(argv[1], "-h") == 0)
{
print_help(executable);
return 0;
}
else if (argc != 3)
{
printf("Error: invalid arguments\n\n");
print_help(executable);
return 1;
}
char *n_str = argv[1];
char *thread_count_str = argv[2];
int n = atoi(n_str);
int thread_count = atoi(thread_count_str);
if (n <= 2)
{
printf("Error: invalid N value \"%s\"\n", n_str);
return 1;
}
if (thread_count <= 0 || thread_count > MAX_THREAD_NUM)
{
printf("Error: invalid thread count \"%s\"\n", thread_count_str);
return 1;
}
int *primes = (int *)malloc((n + 1) * sizeof(int));
primes[0] = 0;
primes[1] = 0;
for (int i = 2; i <= n; i++)
{
primes[i] = 1;
}
double start_time, finish_time;
start_time = omp_get_wtime(); 
for (int i = 2; i * i <= n; i++)
{
if (primes[i] == 0)
{
continue;
}
#pragma omp parallel for num_threads(thread_count)
for (int j = i * i; j <= n; j += i)
{
if (primes[j] == 1)
{
primes[j] = 0;
}
}
}
finish_time = omp_get_wtime(); 
printf("Time take for the main part: %f\n", finish_time - start_time);
char *filename = (char *)malloc(strlen(n_str) + 4);
strcpy(filename, n_str);
strcat(filename, ".txt");
FILE *fp = fopen(filename, "w");
if (fp == NULL)
{
printf("Error: cannot open file \"%s\"\n", filename);
return 1;
}
int order = 1;
for (int i = 2; i <= n; i++)
{
if (primes[i] == 0)
{
continue;
}
fprintf(fp, "%d %d\n", order, i);
order += 1;
}
return 0;
}
void print_help(char *executable)
{
printf("usage: %s N t\n\n", executable);
printf("A parallel prime number generator written in OpenMP.\n\n");
printf("positional arguments:\n");
printf("  N          the ceiling for generated prime numbers, N > 2 \n");
printf("  t          the number of threads, 0 < t <= %d\n", MAX_THREAD_NUM);
}
