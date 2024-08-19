#include <stdlib.h>
int main(int argc, char * argv[])
{
int i;
int len = 100;
int a[len], b[len];
int _ret_val_0;
if (argc>1)
{
len=atoi(argv[1]);
}
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=i;
b[i]=(i+1);
}
#pragma cetus private(i) 
#pragma loop name main#1 
for (i=0; i<(len-1); i ++ )
{
a[i+1]=(a[i]*b[i]);
}
#pragma cetus private(i) 
#pragma loop name main#2 
for (i=0; i<len; i ++ )
{
printf("i=%d a[%d]=%d\n", i, i, a[i]);
}
_ret_val_0=0;
return _ret_val_0;
}
