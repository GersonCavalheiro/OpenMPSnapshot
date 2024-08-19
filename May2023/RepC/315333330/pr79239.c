#pragma GCC push_options
#pragma GCC target "fpu=crypto-neon-fp-armv8"
int a, b;
extern __inline __attribute__((__gnu_inline__)) int fn1() {}
#pragma GCC pop_options
void
fn2() {
if (b * 0.77 + 0.5)
a = 0;
}
