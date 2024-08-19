#pragma GCC push_options
#pragma GCC target ("lwp")
void
foo (unsigned long x, unsigned int y)
{
__builtin_ia32_lwpval64 (x, y, 1);
}
#pragma GCC pop_options
void
bar (unsigned long x, unsigned int y)
{
__builtin_ia32_lwpval64 (x, y, 1);	 
}
