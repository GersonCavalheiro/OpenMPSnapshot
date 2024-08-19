double a[1000][1000], v[1000], v_out[1000];
int init()
{
int i, j, k;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name init#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<1000; i ++ )
{
#pragma cetus lastprivate(j) 
#pragma loop name init#0#0 
#pragma cetus parallel 
#pragma omp parallel for lastprivate(j)
for (j=0; j<1000; j ++ )
{
a[i][j]=((i*j)+0.01);
}
v_out[i]=((i*j)+0.01);
v[i]=((i*j)+0.01);
}
_ret_val_0=0;
return _ret_val_0;
}
void mv()
{
int i, j;
#pragma cetus private(i, j) 
#pragma loop name mv#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<1000; i ++ )
{
double sum = 0.0;
#pragma cetus private(j) 
#pragma loop name mv#0#0 
#pragma cetus reduction(+: sum) 
#pragma cetus parallel 
#pragma omp parallel for private(j) reduction(+: sum)
for (j=0; j<1000; j ++ )
{
sum+=(a[i][j]*v[j]);
}
v_out[i]=sum;
}
return ;
}
int print()
{
int i, j, k;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name print#0 
for (i=0; i<1000; i ++ )
{
#pragma cetus private(j) 
#pragma loop name print#0#0 
for (j=0; j<1000; j ++ )
{
printf("%lf\n", a[i][j]);
}
printf("%lf\n", v_out[i]);
printf("%lf\n", v[i]);
}
_ret_val_0=0;
return _ret_val_0;
}
int main()
{
int _ret_val_0;
init();
mv();
print();
_ret_val_0=0;
return _ret_val_0;
}
