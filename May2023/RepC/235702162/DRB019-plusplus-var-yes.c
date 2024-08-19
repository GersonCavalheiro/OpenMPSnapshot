#include <stdlib.h>
#include <stdio.h>
int main(int argc, char * argv[])
{
int i;
int inLen = 1000;
int outLen = 0;
int input[inLen];
int output[inLen];
int _ret_val_0;
if (argc>1)
{
inLen=atoi(argv[1]);
}
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
printf("output[0]=%d\n", output[0]);
_ret_val_0=0;
return _ret_val_0;
}
