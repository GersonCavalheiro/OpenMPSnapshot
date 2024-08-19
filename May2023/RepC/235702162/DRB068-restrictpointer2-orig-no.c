#include <stdlib.h>
#include <stdio.h>
void init(int n, int * restrict a, int * restrict b, int * restrict c)
{
int i;
#pragma cetus private(i) 
#pragma loop name init#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<n; i ++ )
{
a[i]=1;
b[i]=i;
c[i]=(i*i);
}
return ;
}
void foo(int n, int * restrict a, int * restrict b, int * restrict c)
{
int i;
#pragma cetus private(i) 
#pragma loop name foo#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<n; i ++ )
{
a[i]=(b[i]+c[i]);
}
return ;
}
void print(int n, int * restrict a, int * restrict b, int * restrict c)
{
int i;
#pragma cetus private(i) 
#pragma loop name print#0 
for (i=0; i<n; i ++ )
{
printf("%d %d %d\n", a[i], b[i], c[i]);
}
return ;
}
int main()
{
int n = 1000;
int * a, * b, * c;
int _ret_val_0;
a=((int * )malloc(n*sizeof (int)));
if (a==0)
{
fprintf(stderr, "skip the execution due to malloc failures.\n");
_ret_val_0=1;
return _ret_val_0;
}
b=((int * )malloc(n*sizeof (int)));
if (b==0)
{
fprintf(stderr, "skip the execution due to malloc failures.\n");
_ret_val_0=1;
return _ret_val_0;
}
c=((int * )malloc(n*sizeof (int)));
if (c==0)
{
fprintf(stderr, "skip the execution due to malloc failures.\n");
_ret_val_0=1;
return _ret_val_0;
}
init(n, a, b, c);
foo(n, a, b, c);
print(n, a, b, c);
free(a);
free(b);
free(c);
_ret_val_0=0;
return _ret_val_0;
}
