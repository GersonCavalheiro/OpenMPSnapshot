#pragma GCC push_options
#pragma GCC target "avx512vl"
#pragma GCC pop_options
#pragma GCC target ""
__attribute__ ((__vector_size__ (32))) long long a;
__attribute__ ((__vector_size__ (16))) int b;
void
foo (const long long *p)
{
volatile __attribute__ ((__vector_size__ (32))) long long c;
c = __builtin_ia32_gather3siv4di (a, p, b, 1, 1);		
}
