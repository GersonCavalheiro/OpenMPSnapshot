#pragma GCC target "+nosve"
void
bic_6 (int *a)
{
for (int i = 0; i < 1024; i++)
a[i] &= ~(0xab);
}
void
bic_7 (int *a)
{
for (int i = 0; i < 1024; i++)
a[i] &= ~(0xcd00);
}
void
bic_8 (int *a)
{
for (int i = 0; i < 1024; i++)
a[i] &= ~(0xef0000);
}
void
bic_9 (int *a)
{
for (int i = 0; i < 1024; i++)
a[i] &= ~(0x12000000);
}
void
bic_10 (short *a)
{
for (int i = 0; i < 1024; i++)
a[i] &= ~(0x34);
}
void
bic_11 (short *a)
{
for (int i = 0; i < 1024; i++)
a[i] &= ~(0x5600);
}
