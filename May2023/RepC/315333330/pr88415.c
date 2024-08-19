void
lx (_Complex int *yn)
{
int mj;
#pragma omp for
for (mj = 0; mj < 1; ++mj)
yn[mj] += 1;
}
