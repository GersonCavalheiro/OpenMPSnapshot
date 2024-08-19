#include <string.h>
int main(int argc, char * argv[])
{
int i;
int j;
double a[20][20];
int _ret_val_0;
memset(a, 0, sizeof a);
#pragma cetus private(i, j) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<20; i+=1)
{
#pragma cetus private(j) 
#pragma loop name main#0#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<20; j+=1)
{
a[i][j]+=((i+j)+0.1);
}
}
#pragma cetus private(i, j) 
#pragma loop name main#1 
for (i=0; i<(20-1); i+=1)
{
#pragma cetus private(j) 
#pragma loop name main#1#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<20; j+=1)
{
a[i][j]+=a[i+1][j];
}
}
#pragma cetus private(i, j) 
#pragma loop name main#2 
for (i=0; i<20; i+=1)
{
#pragma cetus private(j) 
#pragma loop name main#2#0 
for (j=0; j<20; j+=1)
{
printf("%lf\n", a[i][j]);
}
}
_ret_val_0=0;
return _ret_val_0;
}
