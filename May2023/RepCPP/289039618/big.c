#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main()
{
long size = 512 * 5000;

int *data = (int*)malloc(size * sizeof(int));

#pragma omp parallel for 
for( long i = 0; i < 512 * 5000; i++)
{
int rank = omp_get_thread_num();

data[i] = rank;

}

free(data);
return 0;
}
