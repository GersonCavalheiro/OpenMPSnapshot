void foo(int * a, int n, int g)
{
int i;
#pragma cetus private(i) 
#pragma loop name foo#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<n; i ++ )
{
a[i]=(a[i]+g);
}
return ;
}
int a[100];
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
a[i]=i;
}
foo(a, 100, 7);
#pragma cetus private(i) 
#pragma loop name main#1 
for (i=0; i<100; i ++ )
{
printf("%d\n", a[i]);
}
_ret_val_0=0;
return _ret_val_0;
}
