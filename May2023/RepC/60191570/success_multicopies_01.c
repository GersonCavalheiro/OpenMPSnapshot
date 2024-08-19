void foo(float *a, int n)
{
#pragma omp task copy_in({a[i] , i = 0;n})
{
}
#pragma omp taskwait
}
int main(int argc, char *argv[])
{
float w[10];
foo(w, 10);
return 0;
}
