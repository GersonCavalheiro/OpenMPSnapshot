int foo ();
#pragma weak foo
int
__attribute__((weak))
foo ()
{
return 0;
}
