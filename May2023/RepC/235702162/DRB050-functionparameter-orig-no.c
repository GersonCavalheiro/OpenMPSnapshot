void foo1(double o1[], double c[], int len)
{
int i;
#pragma cetus private() 
#pragma loop name foo1#0 
#pragma cetus parallel 
#pragma omp parallel for
for (i=0; i<len;  ++ i)
{
double volnew_o8 = 0.5*c[i];
o1[i]=volnew_o8;
}
return ;
}
double o1[100];
double c[100];
int main()
{
int i;
int _ret_val_0;
#pragma cetus private(i) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i)
for (i=0; i<100;  ++ i)
{
c[i]=(i+1.01);
o1[i]=(i+1.01);
}
foo1(o1, c, 100);
#pragma cetus private(i) 
#pragma loop name main#1 
for (i=0; i<100;  ++ i)
{
printf("%lf\n", o1[i]);
}
_ret_val_0=0;
return _ret_val_0;
}
