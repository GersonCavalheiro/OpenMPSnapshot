#include <stdio.h>
#include <unistd.h>
#define N  4
void foo()
{
int diag;
#pragma analysis_check assert induction_var(diag:0:6:1)
for (diag = 0; diag < 2 * N - 1; diag++) {
int j, i;
if (diag<N)
i = diag;
else
i = N-1;
#pragma analysis_check assert induction_var(i:0:diag,4-1:-1;j:diag-i:3:1)
for (j = diag - i; i >= 0 && j < N; i--, j++) 
{}
}
}