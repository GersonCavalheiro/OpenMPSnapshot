#include "stdlib.h"
const int N = 5;
int main()
{
int *a = (int*) malloc(sizeof(int)*5);
int *b = (int*) calloc(5, sizeof(int));
int i;
for(i=0; i<N; i++)
{
a[i] = b[i] = i;
}
#pragma analysis_check assert upper_exposed(a[5], b) defined(b[0], b[5], a)
{
b[0] = 5;
b[5] = a[5];
a = b;
}
return 0;
}