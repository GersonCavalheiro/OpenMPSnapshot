#ifndef ATTRIBUTE_USED
# define ATTRIBUTE_USED __attribute__((used))
#endif
static int lv1;
extern int Av1a __attribute__((alias ("lv1")));
int *pv1a = &Av1a;
static int lv2;
extern int Av2a __attribute__((alias ("lv2")));
int *pv2a = &lv2;
static int lv3;
extern int Av3a __attribute__((alias ("lv3")));
static int *pv3a ATTRIBUTE_USED = &Av3a;
static int lv4;
extern int Av4a __attribute__((alias ("lv4")));
static int *pv4a = &Av4a;
typedef void ftype(void);
static void lf1(void) {}
extern ftype Af1a __attribute__((alias ("lf1")));
ftype *pf1a = &Af1a;
static void lf2(void) {}
extern ftype Af2a __attribute__((alias ("lf2")));
ftype *pf2a = &Af2a;
static void lf3(void) {}
extern ftype Af3a __attribute__((alias ("lf3")));
static ftype *pf3a ATTRIBUTE_USED = &Af3a;
static void lf4(void) {}
extern ftype Af4a __attribute__((alias ("lf4")));
static ftype *pf4a = &Af4a;
int
main() {
#ifdef __mips
asm volatile ("lw $2,%0\n\tlw $2,%1" : : "m" (pv4a), "m" (pf4a) : "$2");
#else
asm volatile ("" : : "m" (pv4a), "m" (pf4a));
#endif
}
