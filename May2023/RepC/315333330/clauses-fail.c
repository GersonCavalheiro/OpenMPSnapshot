void
f (void)
{
int i;
#pragma acc parallel one 
;
#pragma acc kernels eins 
;
#pragma acc data two 
;
#pragma acc parallel
#pragma acc loop deux 
for (i = 0; i < 2; ++i)
;
}
void
f2 (void)
{
int a, b[100];
#pragma acc parallel firstprivate (b[10:20]) 
;
}
