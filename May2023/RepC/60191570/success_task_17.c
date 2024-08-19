#include <stdlib.h>
int b[10];
int main(int argc, char *argv[])
{
size_t dim = 10;
int (*a)[dim] = &b;
#pragma omp task inout(*a)
{
int i = 1;
int elem = (*a)[i];
}
#pragma omp taskwait
return 0;
}
