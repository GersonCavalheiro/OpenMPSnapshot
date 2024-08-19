extern void bar (void);
void
foo (void)
{
}
#pragma interrupt
void
( __attribute__ ((nosave_low_regs)) isr) (void)
{
bar ();
}
void
delay (int a)
{
}
