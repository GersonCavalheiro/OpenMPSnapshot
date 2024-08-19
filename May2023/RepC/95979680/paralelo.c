#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../timer.h"
const int n = 1024ull * 1024ull * 256ull;
const int max_value = 256;
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
printf("%d\n",data[i]);
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
void countHistogram (int *data, int *count, long thread_count)
{
int i;
int local_n = n / thread_count;
#pragma omp parallel num_threads(thread_count) 
{
int *local_count = (int*)calloc(max_value,sizeof(int));
int start_i = omp_get_thread_num()*local_n;
int last_i = start_i + local_n;
#pragma omp for nowait
for (i = start_i; i < last_i; i++)
local_count[data[i]]++;
for (i = 0; i < max_value; i++)
{
#pragma omp atomic
count[i] += local_count[i];
}
free(local_count);
}
}
int main (int argc, char *argv[])
{
int *data;							
int *count;
long thread_count;				
double start, finish, elapsed;
thread_count = strtol(argv[1],NULL,0);
data = allocMemory(n);
count = allocMemory(max_value);
generateData(data);
#ifdef DEBUG
printData(data);
#endif
GET_TIME(start);
countHistogram(data,count,thread_count);
GET_TIME(finish);
#ifdef DEBUG
printHistogram(count);
#endif
elapsed = finish - start;
printf("With n = %d\n",n);
printf("Time elapsed = %.10lf s\n",elapsed);
writeHistogram(count);
free(data);
free(count);
return 0;
}
