#pragma omp target device(smp, cuda)
typedef int kaka_t;
int main()
{
#pragma omp task
{}
}
