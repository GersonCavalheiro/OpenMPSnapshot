#include <stdlib.h>
static unsigned long lv1 = 0xdeadbeefUL;
#pragma weak Av1a = lv1
extern unsigned long Av1a;
static unsigned long lf1(void) { return 0x510bea7UL; }
#pragma weak Af1a = lf1
extern unsigned long Af1a(void);
int main (void) {
if (! &Av1a
|| ! &Af1a
|| Av1a != 0xdeadbeefUL
|| Af1a() != 0x510bea7UL)
abort ();
exit (0);
}
