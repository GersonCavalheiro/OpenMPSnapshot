#include <cassert>
int main()
{
int i, k;
unsigned int counter = 0;
#pragma hlt collapse(3)
for (i = -10; i < 10; i += 6)
{
for (int j = 10; j < 0; --j)
{
for (k = 10; k < -10; k -= 6)
{
counter++;
}
}
}
assert(i == 14 && k == -14);
}
