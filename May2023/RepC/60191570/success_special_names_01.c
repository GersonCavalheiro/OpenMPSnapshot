#include <stdio.h>
int main(int argc, char* argv[])
{
#pragma omp task
{
printf("%s\n", __func__);
}
#pragma omp taskwait
}
