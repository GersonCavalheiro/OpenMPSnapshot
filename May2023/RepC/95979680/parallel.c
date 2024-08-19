#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../timer.h"
const int n = 1024ull * 1024ull * 256ull;
const int max_value = 4;
void error (const char *msg)
{
printf("%s\n",msg);
exit(EXIT_FAILURE);
}
int* allocMemory (int size)
{
int *buffer = (int*)calloc(size,sizeof(int));
if (buffer == NULL)
error("[-] ERROR! Cannot allocate memory!");
return buffer;
}
void generateData (int *data)
{
int i;
for (i = 0; i < n; i++)
data[i] = rand() % max_value;
}
void printData (int *data)
{
int i;
for (i = 0; i < n; i++)
printf("%d ",data[i]);
printf("\n");
}
void printHistogram (int *count)
{
int i;
for (i = 0; i < max_value; i++)
printf("%d = %d\n",i,count[i]);
}
void writeHistogram (int *count)
{
FILE *out = fopen("count.dat","w+");
int i;
for (i = 0; i < max_value; i++)
fprintf(out,"%d %d\n",i,count[i]);
fclose(out);
}
void countHistogram (int *data, int *count)
{
int i;
for (i = 0; i < n; i++)
count[data[i]]++;
}
int* countLocalHistogram (int data[])
{
int i;
int local_n = n / omp_get_num_threads();
int start_i = omp_get_thread_num()*local_n;
int finish_i = start_i + local_n;
int *local_count = allocMemory(max_value);
for (i = start_i; i < finish_i; i++)
local_count[data[i]]++;
return local_count;
}
void Usage (const char program_name[])
{
printf("===========================================\n");
printf("Usage:> %s <num_threads>\n",program_name);
printf("===========================================\n");
}
int main (int argc, char *argv[])
{
int i;
int *data, *count, *local_count;
int num_threads;
double start, finish, elapsed;
if (argc-1 < 1)
{
Usage(argv[0]);
exit(EXIT_FAILURE);
}
num_threads = atoi(argv[1]);
omp_set_num_threads(num_threads);
data = allocMemory(n);
count = allocMemory(max_value);
generateData(data);
GET_TIME(start);
#pragma omp parallel
{
local_count = countLocalHistogram(data);
#pragma omp critical
{
for (i = 0; i < max_value; i++)
count[i] += local_count[i];
}
}
GET_TIME(finish);
elapsed = finish - start;
printf("With n = %d\n",n);
printf("Time elapsed = %.10lf s\n",elapsed);
printHistogram(count);
free(data);
free(count);
return 0;
}
