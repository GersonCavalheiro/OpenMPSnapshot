#include <omp.h>
#include <stdio.h>
#include <limits.h>
int main() {
int sum;
#pragma omp parallel for num_threads(4)
for(int i = 0; i < INT_MAX; i++){}
return 0;
}
