int omp_get_num_threads(void);
int omp_get_thread_num(void);
enum A
{
L=1,
D=2
};
#pragma omp declare reduction (foo: enum A : omp_out=omp_in) initializer(omp_priv = L)
int main (int argc, char* argv[])
{
enum A a;
#ifdef NANOX
#pragma omp for reduction(foo: a)
for(int i=0; i<1; i++)
#else
#pragma omp parallel reduction(foo: a)
#endif
{}
return 0;
}
