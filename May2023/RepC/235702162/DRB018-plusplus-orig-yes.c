#include <stdlib.h>
#include <stdio.h>
int input[1000];
int output[1000];
int main()
{
int i;
int inLen = 1000;
int outLen = 0;
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<inLen;  ++ i)
{
input[i]=i;
}
#pragma cetus private(i) 
#pragma loop name main#1 
for (i=0; i<inLen;  ++ i)
{
output[outLen ++ ]=input[i];
}
printf("output[500]=%d\n", output[500]);
_ret_val_0=0;
return _ret_val_0;
}
