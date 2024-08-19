#include <stdlib.h>
int main(int argc, char * argv[])
{
int i;
int len = 100;
int numNodes = len, numNodes2 = 0;
int x[len];
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
if ((i%2)==0)
{
x[i]=5;
}
else
{
x[i]=( - 5);
}
}
#pragma cetus private(i) 
#pragma loop name main#1 
#pragma cetus reduction(+: numNodes2) 
#pragma cetus parallel 
#pragma omp parallel for private(i) reduction(+: numNodes2)
for (i=(numNodes-1); i>( - 1);  -- i)
{
if (x[i]<=0)
{
numNodes2 -- ;
}
}
printf("numNodes2 = %d\n", numNodes2);
_ret_val_0=0;
return _ret_val_0;
}
