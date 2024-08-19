#include "omp.h"
#include <iostream>
#include "submit.h"
using namespace std;

void vec_random_norm(float* rand_arr, int n)
{
float min_elem, max_elem;

#pragma omp single copyprivate(min_elem, max_elem)
{
min_elem = rand_arr[0];
max_elem = rand_arr[0];

for (int i = 1; i < n; i++) {
if (rand_arr[i] < min_elem) min_elem = rand_arr[i];
if (rand_arr[i] > max_elem) max_elem = rand_arr[i];
}
}

int index = omp_get_thread_num();

rand_arr[index] = (rand_arr[index] - min_elem) / (max_elem - min_elem);

Submit_test();
}