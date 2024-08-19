#pragma GCC push_options
#pragma GCC target ("sse")
void
foo (void)
{
__builtin_ia32_pause ();
}
#pragma GCC pop_options
void
bar (void)
{
__builtin_ia32_pause ();
}
