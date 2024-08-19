int main(int argc, char *argv[])
{
#pragma omp task
{
}
#pragma omp taskwait
return 0;
}
