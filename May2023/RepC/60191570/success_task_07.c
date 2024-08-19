#include <stdlib.h>
typedef struct
{
int nz;
int diag;
} X;
int main(void)
{
int i;
X foo;
foo.diag = 3;
int bar[40];
#pragma omp task inout(bar[0:foo.diag])
for(i=0;i<foo.diag;i++)
{
bar[i] = i;
}
#pragma omp taskwait
if (bar[0] != 0) abort();
if (bar[1] != 1) abort();
if (bar[2] != 2) abort();
return 0;
}
