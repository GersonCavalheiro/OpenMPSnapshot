#pragma GCC target "+nosve"
extern void abort (void);
unsigned char in[8] __attribute__((__aligned__(16)));
int
main (unsigned char argc, char **argv)
{
unsigned char i = 0;
unsigned char sum = 1;
for (i = 0; i < 8; i++)
in[i] = (i + i + 1) & 0xfd;
asm volatile ("" : : : "memory");
for (i = 0; i < 8; i++)
sum |= in[i];
if (sum != 13)
{
__builtin_printf ("Failed %d\n", sum);
abort ();
}
return 0;
}
