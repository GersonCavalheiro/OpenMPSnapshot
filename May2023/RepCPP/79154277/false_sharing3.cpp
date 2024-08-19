
#include <cstdlib>

#include <omp.h>
#define MAX_NUM_THREADS 64


int main(int argc, char ** argv)
{
long DIM = 10000;

int odds=0;
int result[MAX_NUM_THREADS];
int *matrix = new int[DIM*DIM]; 
for (long i=0; i<DIM*DIM; ++i)
matrix[i] = rand();

#pragma omp parallel
{
int p = omp_get_thread_num();
int local = 0;
result[p] = 0;
#pragma omp for
for( long i = 0; i < DIM; ++i )
{
for (long k = 0; k<100; ++k)
for( long j = 0; j < DIM; ++j )
if ( matrix[i*DIM + j] % 2 != 0 )
{
if (p%2) 
++result[p];
else
local += result[p];
}
}

#pragma omp atomic
odds += result[p];
}



}
