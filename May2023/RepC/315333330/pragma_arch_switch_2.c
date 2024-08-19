#pragma GCC target ("arch=armv6")
int test_assembly (int hi, int lo)
{
int res;
__asm__ __volatile__ (
"uxtah   %0, %1, %2\n"
: "=r" (res)
: "r" (hi),  "r" (lo));
return res;
}
