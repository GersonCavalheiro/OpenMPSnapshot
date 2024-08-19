#include <stdlib.h>
int main(int argc, char *argv[])
{
char a1 = 0;
char a2 = 0;
char a3 = 0;
#pragma omp parallel sections shared(a1, a2, a3)
{
#pragma omp section
{
a1 = 1;
}
#pragma omp section
{
a2 = 1;
}
#pragma omp section
{
a3 = 1;
}
}
if (!a1 || !a2 || !a3)
abort();
return 0;
}
