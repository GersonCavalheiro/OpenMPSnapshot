struct S { int x; } s;
template<typename T> void f1()
{
#pragma omp atomic
s += 1;		
}
template<typename T> void f2(float *f)
{
#pragma omp atomic
*f |= 1;		
}
template<typename T> void f3(float *f)
{
#pragma omp atomic
*f |= sizeof (T);	
}
template<typename T> void f4(T *t)
{
#pragma omp atomic
*t += 1;
}
template<typename T> void f5(float *f)
{
#pragma omp atomic
*f |= (T)sizeof(T);	
}
