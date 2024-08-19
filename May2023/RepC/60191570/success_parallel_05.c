#include <stdlib.h>
int a[10];
int main(int argc, char* argv[])
{
int b[10];
a[3] = 4;
b[6] = 9;
#pragma omp parallel shared(a, b)
{
a[2] = a[3] + 4;
b[5] = b[6] + 7;
}
if (a[2] != 8)
abort();
if (b[5] != 16)
abort();
return 0;
}
