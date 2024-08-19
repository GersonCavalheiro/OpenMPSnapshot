int bar (int);
double bar (double);
template <typename T>
void
foo (T x)
{
#pragma omp for collapse (x + 1) 
for (int i = 0; i < 10; i++)
;
#pragma omp for ordered (x + 1) 
for (int i = 0; i < 10; i++)
for (int j = 0; j < 10; j++)
;
}
