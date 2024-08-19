#pragma GCC target "+nosve"
void
orr_0 (int *a)
{
for (int i = 0; i < 1024; i++)
a[i] |= 0xab;
}
void
orr_1 (int *a)
{
for (int i = 0; i < 1024; i++)
a[i] |= 0x0000cd00;
}
void
orr_2 (int *a)
{
for (int i = 0; i < 1024; i++)
a[i] |= 0x00ef0000;
}
void
orr_3 (int *a)
{
for (int i = 0; i < 1024; i++)
a[i] |= 0x12000000;
}
void
orr_4 (short *a)
{
for (int i = 0; i < 1024; i++)
a[i] |= 0x00340034;
}
void
orr_5 (int *a)
{
for (int i = 0; i < 1024; i++)
a[i] |= 0x56005600;
}
