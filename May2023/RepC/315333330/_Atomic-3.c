void
f1 (void)
{
_Atomic int i = 0, k[4];
int j = 0;
k[0] = 0;
k[1] = 0;
k[2] = 0;
k[3] = 0;
#pragma omp parallel reduction (+:i)		
i++;
#pragma omp declare reduction (foo: _Atomic int: omp_out += omp_in) initializer (omp_priv = omp_orig * 0)	
#pragma omp declare reduction (bar: int: omp_out += omp_in) initializer (omp_priv = omp_orig * 0)
#pragma omp parallel reduction (bar:j)
j++;
#pragma omp parallel reduction (bar:i)	
i++;
#pragma omp parallel reduction (+:k)		
k[1]++;
#pragma omp parallel reduction (+:k[1:2])	
k[1]++;
}
void
f2 (int *_Atomic p)
{
#pragma omp simd aligned (p : 16)		
for (int i = 0; i < 16; i++)
p[i]++;
}
_Atomic int x;
void
f3 (_Atomic int *p)
{
int i;
#pragma omp atomic write
x = 6;					
#pragma omp atomic read
i = x;					
#pragma omp atomic update
x += 6;					
#pragma omp atomic capture
i = x *= 2;					
#pragma omp atomic write
p[2] = 6;					
#pragma omp atomic read
i = p[2];					
#pragma omp atomic update
p[2] += 6;					
#pragma omp atomic capture
i = p[2] *= 2;				
}
#pragma omp declare simd linear(x:1)		
int
f4 (_Atomic int x, int y)
{
return x + y;
}
