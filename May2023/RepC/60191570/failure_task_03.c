#pragma omp task inout(*x, *y)
void g(int *y)
{
int *x;
}
