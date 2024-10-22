#pragma GCC target "+nosve"
extern void abort (void);
#define N 16
static void
movi_msl8 (int *__restrict a)
{
int i;
for (i = 0; i < N; i++)
a[i] = 0xabff;
}
static void
movi_msl16 (int *__restrict a)
{
int i;
for (i = 0; i < N; i++)
a[i] = 0xabffff;
}
static void
mvni_msl8 (int *__restrict a)
{
int i;
for (i = 0; i < N; i++)
a[i] = 0xffff5400;
}
static void
mvni_msl16 (int *__restrict a)
{
int i;
for (i = 0; i < N; i++)
a[i] = 0xff540000;
}
static void
movi_float_lsl24 (float * a)
{
int i;
for (i = 0; i < N; i++)
a[i] = 128.0;
}
int
main (void)
{
int a[N] = { 0 };
float b[N] = { 0 };
int i;
#define CHECK_ARRAY(a, val)	\
for (i = 0; i < N; i++)	\
if (a[i] != val)		\
abort ();
movi_msl8 (a);
CHECK_ARRAY (a, 0xabff);
movi_msl16 (a);
CHECK_ARRAY (a, 0xabffff);
mvni_msl8 (a);
CHECK_ARRAY (a, 0xffff5400);
mvni_msl16 (a);
CHECK_ARRAY (a, 0xff540000);
movi_float_lsl24 (b);
CHECK_ARRAY (b, 128.0);
return 0;
}
