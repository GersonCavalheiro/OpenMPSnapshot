#pragma GCC push_options
#pragma GCC target ("avx512vl,avx512bw")
void
foo (char *x, char __attribute__ ((__vector_size__(32))) *y, int z)
{
__builtin_ia32_storedquqi256_mask (x, *y, z);
}
#pragma GCC pop_options
void
bar (char *x, char __attribute__ ((__vector_size__(32))) *y, int z)
{
__builtin_ia32_storedquqi256_mask (x, *y, z); 
}
