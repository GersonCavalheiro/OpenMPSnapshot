void
fn3 (int x)
{
double b[3 * x];
int i;
#pragma omp target
#pragma omp parallel for
for (i = 0; i < x; i++)
b[i] += 1;
}
