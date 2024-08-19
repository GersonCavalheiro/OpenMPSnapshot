extern void foo (void);
#pragma interrupt
#pragma nosave_low_regs
void
isr (void)
{
foo ();
}
