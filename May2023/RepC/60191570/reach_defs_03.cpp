#include <stddef.h>
#define N 12
int B[10];
void foo(void)
{
int i;
#pragma omp task
{
i = 0;
#pragma analysis_check assert reaching_definition_in(i:0) induction_var(i:0:11:1)
for (; i < N; ++i)
{
#pragma omp task
B[i] += 2;
}
}
#pragma omp taskwait
}
