#include <stdlib.h>
int a;
int main(int argc, char* argv[])
{
int b;
a = 10;
b = 20;
#pragma omp parallel private(a, b)
{
a = a + 3;
b = b + 4;
}
if (a != 10)
abort();
if (b != 20)
abort();
return 0;
}
