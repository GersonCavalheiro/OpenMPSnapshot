#include <stdlib.h>
#include <stdio.h>
void foo(int n, int * restrict  a, int * restrict b, int * restrict  c)
{
int i;
#pragma omp parallel for 
for (i = 0; i < n; i++)
a[i] = b[i] + c[i];  
}
int main()
{
int n = 1000;
int * a , *b, *c;
a = (int*) malloc (n* sizeof (int));
if (a ==0)
{
fprintf (stderr, "skip the execution due to malloc failures.\n");
return 1;
}
b = (int*) malloc (n* sizeof (int));
if (b ==0)
{
fprintf (stderr, "skip the execution due to malloc failures.\n");
return 1;
}
c = (int*) malloc (n* sizeof (int));
if (c ==0)
{
fprintf (stderr, "skip the execution due to malloc failures.\n");
return 1;
}
foo (n, a, b,c);
free (a);
free (b);
free (c);
return 0;
}  
