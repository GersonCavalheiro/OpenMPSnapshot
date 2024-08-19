extern void foo (void);
#pragma interrupt
void
isr1 (void)
{
foo ();
}
#pragma interrupt
void
isr2 (void)
{
foo ();
}
