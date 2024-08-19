int main()
{
double d1 = 0.0, d2 = 0.0;
#pragma omp task shared(d1, d2)
{
#pragma omp atomic
d1 += d2;
}
#pragma omp taskwait
}
