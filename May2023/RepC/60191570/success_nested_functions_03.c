#include <assert.h>
int main_wd = 0;
int main(int argc, char *argv[])
{
int y;
main_wd = nanos_get_wd_id(nanos_current_wd());
#pragma omp target device(smp) no_copy_deps
#pragma omp task inout(*x)
void g(int *x);
y = 1;
g(&y);
g(&y);
g(&y);
g(&y);
g(&y);
#pragma omp taskwait
return 0;
}
void g(int *x)
{
int local_wd = nanos_get_wd_id(nanos_current_wd());
assert(local_wd != main_wd);
(*x)++;
}
