#include <stdio.h>
#include <omp.h>

int main()
{
int foo = 0;
int bar = 0;

#pragma omp parallel reduction(max:foo,bar)
{
foo = omp_get_thread_num();
bar = omp_get_num_threads();
}

printf("Foo: %d\nBar: %d\n", foo, bar);
}
