#include <stdlib.h>
#include "omp.h"
int main(int argc, char* argv[])
{
int i = 3;
#pragma omp atomic
i += 2;
#pragma omp atomic
i++;
if (i != 6)
abort();
return 0;
}
