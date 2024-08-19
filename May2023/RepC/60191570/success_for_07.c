#include <stdlib.h>
int a[10];
int main(int argc, char *argv[])
{
int b[10];
int i;
a[1] = 3;
a[2] = 9;
b[4] = 4;
b[5] = 10;
#pragma omp for private(a, b)
for (i = 0; i < 10; i++)
{
a[1] = a[2] + 3;
b[4] = b[5] + 6;
}
if (a[1] != 3) 
abort();
if (a[2] != 9) 
abort();
if (b[4] != 4) 
abort();
if (b[5] != 10) 
abort();
return 0;
}
