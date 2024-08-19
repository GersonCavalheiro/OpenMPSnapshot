int main(int argc, char * argv[])
{
int i;
int len = 1000;
int a[len];
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=i;
}
#pragma cetus private(i) 
#pragma loop name main#1 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<len; i ++ )
{
a[i]=(a[i]+1);
}
#pragma cetus private(i) 
#pragma loop name main#2 
for (i=0; i<len; i ++ )
{
printf("%d", a[i]);
}
_ret_val_0=0;
return _ret_val_0;
}
