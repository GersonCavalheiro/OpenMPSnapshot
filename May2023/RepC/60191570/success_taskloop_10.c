#include<assert.h>
int main() {
#pragma omp taskloop grainsize(1)
for (unsigned int x = 0u; x < 0u; ++x)
{
assert(0);
}
}
