#include <stdio.h>
#include <stdlib.h>
int main(int argc, char * argv[])
{
int i, x;
int len = 10000;
int _ret_val_0;
if (argc>1)
{
len=atoi(argv[1]);
}
#pragma cetus private(i) 
#pragma cetus lastprivate(x) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i) lastprivate(x)
for (i=0; i<len; i ++ )
{
x=i;
}
printf("x=%d", x);
_ret_val_0=0;
return _ret_val_0;
}
