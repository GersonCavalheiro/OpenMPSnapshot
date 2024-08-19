#pragma GCC target("mvcle")
void p1(char *b)
{
__builtin_memset (b, 0, 400);
}
#pragma GCC reset_options
__attribute__ ((target("mvcle")))
void a1(char *b)
{
__builtin_memset (b, 0, 400);
}
