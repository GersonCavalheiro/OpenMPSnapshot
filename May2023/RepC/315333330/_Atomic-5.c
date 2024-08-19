void
f1 (void)
{
struct S { int a; int b[2]; _Atomic int *c; };
_Atomic int a = 0, b[2];
_Atomic int d[3];
_Atomic struct S c = (struct S) { 3, { 4, 5 }, d };
int *_Atomic p;
_Atomic int *q;
int e[3] = { 1, 2, 3 };
b[0] = 1;
b[1] = 2;
d[0] = 6;
d[1] = 7;
d[2] = 8;
p = e;
#pragma omp target map(tofrom: a)		
;
#pragma omp target map(to: b)			
;
#pragma omp target map(from: b[1:1])		
;
#pragma omp target map(to: c.a)		
;
#pragma omp target map(to: c.b[1])		
;
#pragma omp target data map(c)		
{
#pragma omp target update to (c.a)		
#pragma omp target update from (c.b[1])	
#pragma omp target update to (c)		
}
#pragma omp target map(to: c.c[0:])		
;
#pragma omp target map(to: p[1:2])		
;
#pragma omp target map(to: q[1:2])		
;
}
void
f2 (void)
{
_Atomic int a = 0, b[2] = { 1, 2 };
#pragma omp target defaultmap(tofrom:scalar)	
a++;
#pragma omp target				
b[0]++;
}
void
f3 (void)
{
_Atomic int a = 0, b[2] = { 1, 2 };
#pragma omp target				
a++;
#pragma omp target firstprivate (a)		
a++;
#pragma omp target firstprivate (b)		
b[0]++;
}
