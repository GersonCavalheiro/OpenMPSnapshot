#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include "omp.h"


const int A = 1000000;
const int P = 4;

int main()
{
int start_time = clock();





int count = 0;
bool isDiv[A];

#pragma omp parallel num_threads(P) default(none) shared(A, isDiv) reduction(+:count)
{
int task_id = omp_get_thread_num();
int H = A / P;
int start_index = task_id * H;
int end_index;
if (task_id == P - 1) 			
end_index = A - 1; 			
else							
end_index = start_index + H - 1; 
if (start_index == 0) start_index=1;
for (int i = start_index; i <= end_index; i++)
{
if (A % i == 0)
{
isDiv[i] = true;
count++;
}
}
}

printf("Count: %d\n", count);

int* res;
res = (int*)malloc(sizeof(int) * count);
count = 0;
for (int i = 1; i < A; i++)
{
if (isDiv[i] == true)
{
res[count] = i;
count++;
}
}
for (int i = 0; i < count; i++)
{
printf("%d ", res[i]);
}

int end_time = clock();

double time = (double)(end_time - start_time) / 1000;
printf("Time: %f ms.\n", time);

return 0;
}
