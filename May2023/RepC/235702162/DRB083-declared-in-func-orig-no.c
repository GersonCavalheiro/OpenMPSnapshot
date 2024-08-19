void foo()
{
int q=0; 
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
