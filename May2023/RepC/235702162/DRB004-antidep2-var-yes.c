#include <stdlib.h>
int main(int argc, char * argv[])
{
int i, j;
int len = 20;
double a[len][len];
int _ret_val_0;
if (argc>1)
{
len=atoi(argv[1]);
}
#pragma cetus private(i, j) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<len; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#0#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<len; j ++ )
{
a[i][j]=0.5;
}
}
#pragma cetus private(i, j) 
#pragma loop name main#1 
for (i=0; i<(len-1); i+=1)
{
#pragma cetus private(j) 
#pragma loop name main#1#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<len; j+=1)
{
a[i][j]+=a[i+1][j];
}
}
#pragma cetus private(i, j) 
#pragma loop name main#2 
for (i=0; i<len; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#2#0 
for (j=0; j<len; j ++ )
{
printf("%lf\n", a[i][j]);
}
}
_ret_val_0=0;
return _ret_val_0;
}
