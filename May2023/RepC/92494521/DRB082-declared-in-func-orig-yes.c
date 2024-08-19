void foo()
{
static int q; 
q += 1;
}
int main()
{ 
#pragma omp parallel 
{
foo();
}
return 0;   
}
