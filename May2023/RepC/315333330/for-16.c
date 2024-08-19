template<typename T>
void foo ()
{
#pragma omp for
for (unsigned int i = 0; i < 10; i++);
#pragma omp for
for (int j = 0; ; j++); 
#pragma omp for
for (int k = 0; k == 1; k++); 
#pragma omp for
for (int l = 0; l < 10; ); 
#pragma omp for
for (int m = 0; m < 10; m *= 3); 
#pragma omp for
for (T n = 0; ; n++); 
#pragma omp for
for (T o = 0; o == 1; o++); 
#pragma omp for
for (T p = 0; p < 10; ); 
#pragma omp for
for (T q = 0; q < 10; q *= 3); 
}
void bar ()
{
#pragma omp for
for (int m = 0; m < 10; m *= 3); 
}
