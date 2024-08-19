#if defined(__i386__) || defined(__x86_64__)
#pragma GCC target "fpmath=sse"
#endif
static inline __attribute__ ((always_inline)) int
foo (int x)
{
return x;
}
int
bar (int x)
{
return foo (x);
}
