extern void foo(void);
#pragma GCC target("packed-stack")
int p1(void)
{
foo();
return 1;
}
#pragma GCC reset_options
#pragma GCC target("no-packed-stack")
int p0(void)
{
foo();
return 2;
}
int p0b(void)
{
foo();
return 2;
}
#pragma GCC reset_options
__attribute__ ((target("packed-stack")))
int a1(void)
{
foo();
return 4;
}
__attribute__ ((target("packed-stack")))
int a1b(void)
{
foo();
return 4;
}
__attribute__ ((target("packed-stack")))
int a1c(void)
{
foo();
return 4;
}
__attribute__ ((target("packed-stack")))
int a1d(void)
{
foo();
return 4;
}
__attribute__ ((target("no-packed-stack")))
int a0(void)
{
foo();
return 8;
}
__attribute__ ((target("no-packed-stack")))
int a0b(void)
{
foo();
return 8;
}
__attribute__ ((target("no-packed-stack")))
int a0c(void)
{
foo();
return 8;
}
__attribute__ ((target("no-packed-stack")))
int a0d(void)
{
foo();
return 8;
}
__attribute__ ((target("no-packed-stack")))
int a0e(void)
{
foo();
return 8;
}
__attribute__ ((target("no-packed-stack")))
int a0f(void)
{
foo();
return 8;
}
__attribute__ ((target("no-packed-stack")))
int a0g(void)
{
foo();
return 8;
}
__attribute__ ((target("no-packed-stack")))
int a0h(void)
{
foo();
return 8;
}
