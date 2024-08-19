
#include "bits/stdc++.h"
#include "omp.h"
using namespace std;
int main()
{
omp_set_num_threads(5);

#pragma omp parallel
{
printf("Hello World\n");
}
}
