#pragma weak _Z3foov
int
main (void)
{
extern int foo (void);
if (&foo)
return foo ();
return 0;
}
