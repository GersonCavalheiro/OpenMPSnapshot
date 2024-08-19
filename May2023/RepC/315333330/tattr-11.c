#pragma GCC target("no-mvcle")
void p0(char *b)
{
__builtin_memset (b, 0, 400);
}
#pragma GCC reset_options
__attribute__ ((target("no-mvcle")))
void a0(char *b)
{
__builtin_memset (b, 0, 400);
}
void d(char *b)
{
__builtin_memset (b, 0, 400);
}
