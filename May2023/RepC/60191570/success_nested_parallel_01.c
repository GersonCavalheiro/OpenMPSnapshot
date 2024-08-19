int main(int argc, char* argv[])
{
int k = 1;
#pragma omp parallel private(k)
{
k = 1;
#pragma omp parallel firstprivate(k)
{ 
k = 2;
}
}
return 0;
}
