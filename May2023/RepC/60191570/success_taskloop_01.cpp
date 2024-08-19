#include <cassert>
const int MAX_SIZE = 256, ELEMS = 10;
int main()
{
int test[MAX_SIZE];
int from = 5;
for (int i = 0; i < ELEMS; i++)
{
test[i] = 0;
}
for (int i = ELEMS; i < MAX_SIZE; i++)
{
test[i] = -1;
}
#pragma oss task for  shared(test)
for (int i = from; i < 10; i++)
{
int element = test[i];
assert(element == 0);
}
#pragma oss taskwait
}
