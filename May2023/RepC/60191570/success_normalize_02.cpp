#include <cassert>
int main()
{
int i, j;
unsigned int counter = 0;
#pragma hlt normalize
for (i = -10; i < 10; i += 6)
{
counter++;
}
#pragma hlt normalize
for (j = 10; j < -10; j -= 6)
{
counter++;
}
assert(i == 14 && j == -14);
}
