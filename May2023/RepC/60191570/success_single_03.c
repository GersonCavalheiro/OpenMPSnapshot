#include <assert.h>
int main(int argc, char **argv)
{
int arrSize = 20;
int testArr[arrSize];
int i;
#pragma omp single
{
testArr[0] = -1;
}
#pragma omp single
{
for (i = 1; i < sizeof(testArr)/sizeof(testArr[0]); i++)
{
testArr[i] = i;
}
}
assert(testArr[0] == -1);
for (i = 1; i < arrSize; i++)
{
assert(testArr[i] == i);
}
return 0;
}
