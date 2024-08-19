void
test1 (void)
{
#pragma omp for
for (i = 0; i < 1; ++i); 
}
void
test2 (void)
{
int i;
#pragma omp for
for (i = j; i < 1; ++i); 
}
void
test3 (void)
{
int i;
#pragma omp for
for (i = 0; i < j; ++i); 
}
void
test4 (void)
{
int i;
#pragma omp for
for (i = 0; i < 10; i += j); 
}
