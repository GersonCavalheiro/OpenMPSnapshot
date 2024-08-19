extern void foo (void);
#pragma trapa
void
isr (void)
{
foo ();
}
