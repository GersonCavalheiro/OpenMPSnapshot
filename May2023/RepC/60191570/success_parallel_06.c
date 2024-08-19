#include <stdlib.h>
int a[10];
int main(int argc, char *argv[])
{
int b[10];
a[2] = 3;
b[5] = 4;
#pragma omp parallel private(a, b)
{
a[2] = a[3] + 4;
b[5] = b[6] + 7;
}
if (a[2] != 3)
abort();
if (b[5] != 4)
abort();
return 0;
}
