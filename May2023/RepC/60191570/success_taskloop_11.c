#include<assert.h>
int main()
{
#pragma omp taskloop grainsize(1)
for (unsigned int x = -1u; x <= -1u; ++x)
{
}
}
