int main()
{
int x;
#pragma omp taskwait on(x)
#pragma omp taskwait in(x)
#pragma omp taskwait out(x)
#pragma omp taskwait inout(x)
#pragma omp taskwait depend(in: x)
#pragma omp taskwait depend(out: x)
#pragma omp taskwait depend(inout: x)
}
