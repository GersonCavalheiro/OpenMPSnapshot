struct A
{
int x;
};
void foo ( struct A *a,  struct A *b );
#pragma omp declare reduction ( foo : struct A : foo(&omp_out,&omp_in) ) initializer(omp_priv = {0})
void bar ()
{
#pragma omp declare reduction ( foo : struct A : foo(&omp_out,&omp_in) ) initializer(omp_priv = {0})
}
