#include "submit.h"
#include "omp.h"
#include "stdlib.h"
#include "time.h"
#include "ex_function.h"


float* vec_random(int n, bool normal)
{
float* rand_arr = new float[n];


#pragma omp parallel num_threads(n)
{
int index = omp_get_thread_num();

srand(index);

rand_arr[index] = float(rand() % 100);

if (normal)
{
#pragma omp barrier 
vec_random_norm(rand_arr, n);
}

Submit_test();
}
return rand_arr;
}