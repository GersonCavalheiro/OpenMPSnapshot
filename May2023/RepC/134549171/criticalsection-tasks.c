void crit()
{
#pragma omp task 
{ 
#pragma omp task 
{ 
#pragma omp critical 
{ }       
}
#pragma omp critical 
{
#pragma omp task
{  } 
}
}
}
int main()
{
crit();
return 0;
}
