#include <stdio.h>
void foo(double * a, double * b, int N)
{
int i;
#pragma cetus private(i) 
#pragma loop name foo#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<N; i ++ )
{
b[i]=(a[i]*((double)i));
}
return ;
}
int main(int argc, char * argv[])
{
int i;
int len = 1000;
double a[len], b[len];
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=(((double)i)/2.0);
b[i]=0.0;
}
foo(a, b, len);
printf("b[50]=%f\n", b[50]);
_ret_val_0=0;
return _ret_val_0;
}
