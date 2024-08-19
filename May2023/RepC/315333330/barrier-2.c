void f1(void)
{
#pragma omp barrier a		
}
void f3(bool p)
{
if (p)
#pragma omp barrier		
}				
