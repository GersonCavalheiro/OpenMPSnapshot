int omp_get_num_threads(void);
int omp_get_thread_num(void);
typedef struct {
double real;
double imag;
} my_complex_t;
my_complex_t complex_add (my_complex_t a, my_complex_t b) 
{ 
my_complex_t m = {0.0, 0.0};
return m;
}
my_complex_t complex_mul (my_complex_t a, my_complex_t b)
{ 
my_complex_t m = {0.0, 0.0};
return m;
}
#pragma omp declare reduction(complex_add : my_complex_t : omp_out = complex_add(omp_in, omp_out)) initializer(omp_priv = {0,0})
#pragma omp declare reduction(complex_mul : my_complex_t : omp_out = complex_mul(omp_in, omp_out)) initializer(omp_priv = {1,0})
#define N 100
my_complex_t vector[N];
int foo(my_complex_t x, my_complex_t y)
{
int i;
#pragma omp parallel for reduction(complex_add:x) reduction(complex_mul:y)
for ( i = 0; i < N ; i++ ) 
{
x = complex_add(x,vector[i]);
y = complex_mul(y,vector[i]);
}
return 0;
}
int main(int argc, char* argv[])
{
my_complex_t c = { 0, 0 };
my_complex_t d = { 0, 0 };
return foo(c, d);
}
