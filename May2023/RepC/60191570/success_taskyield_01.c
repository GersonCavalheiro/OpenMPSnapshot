int main(int argc, char*argv[])
{
#pragma omp task
{
#pragma omp taskyield
}
#pragma omp taskwait
return 0;
}
