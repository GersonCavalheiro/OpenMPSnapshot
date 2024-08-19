#include <assert.h>
int x = 42;
int main(int argc, char *argv[])
{
#pragma omp critical
{
x++;
}
#pragma omp critical(A)
{
x++;
}
assert(x == 44);
return 0;
}
