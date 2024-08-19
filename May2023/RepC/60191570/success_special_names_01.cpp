#include <stdio.h>
int main(int argc, char *argv[])
{
#pragma omp task
{
fprintf(stderr, "1 %s\n", __PRETTY_FUNCTION__);
fprintf(stderr, "2 %s\n", __FUNCTION__);
}
#pragma omp taskwait
return 0;
}
