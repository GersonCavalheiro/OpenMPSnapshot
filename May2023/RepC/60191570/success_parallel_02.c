#include <stdlib.h>
int a;
int main(int argc, char *argv[])
{
int b;
a = 20;
b = 30;
#pragma omp parallel shared(a, b)
{
a = a + 3;
b = b + 4;
}
if (a == 20)
abort();
if (b == 30)
abort();
return 0;
}
