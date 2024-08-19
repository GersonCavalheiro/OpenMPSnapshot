#include <stdlib.h>
#include <math.h>
int main(int argc, char *argv[])
{
double d = 1.0;
#pragma omp atomic
d += 3.0;
if (fabs(d - 4.0) > 1e-9)
abort();
return 0;
}
