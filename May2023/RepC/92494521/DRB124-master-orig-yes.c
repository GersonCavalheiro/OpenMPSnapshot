#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
int main (int argc, char **argv)
{
int init, local;
#pragma omp parallel shared(init) private(local)
{
#pragma omp master
{
init = 10;
}
local = init;
}
return 0;
}
