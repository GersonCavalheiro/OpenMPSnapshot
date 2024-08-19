int a[100], b[100], c[100];
int main()
{
int i;
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<100; i ++ )
{
a[i]=(i*40);
b[i]=(i-1);
c[i]=i;
}
#pragma cetus private(i) 
#pragma loop name main#1 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<100; i ++ )
{
a[i]=(b[i]*c[i]);
}
#pragma cetus private(i) 
#pragma loop name main#2 
for (i=0; i<100; i ++ )
{
printf("%d %d %d\n", a[i], b[i], c[i]);
}
_ret_val_0=0;
return _ret_val_0;
}
