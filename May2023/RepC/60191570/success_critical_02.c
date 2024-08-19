#include <stdlib.h>
int main(int argc, char *argv[])
{
unsigned char flagA = 0;
unsigned char flagB = 0;
#pragma omp parallel shared(flagA, flagB)
{
int i;
for (i = 0; i < 100; i++)
{
int j;
for (j = 0; j < 100; j++)
{
#pragma omp critical(A)
{
unsigned char val_of_flagA = j & 0x1;
if (flagA != val_of_flagA)
__builtin_abort();
flagA = (~val_of_flagA) & 0x1;
}
#pragma omp critical(B)
{
unsigned char val_of_flagB = j & 0x1;
if (flagB != val_of_flagB)
__builtin_abort();
flagB = (~val_of_flagB) & 0x1;
}
}
}
}
return 0;
}
