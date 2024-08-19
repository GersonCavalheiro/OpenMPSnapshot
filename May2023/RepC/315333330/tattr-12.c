extern char *bar(int x);
void da(int x)
{
bar(x);
bar(x + 1);
}
#pragma GCC target("stack-size=1024,stack-guard=0")
void p1(int x)
{
bar(x);
bar(x + 1);
}
#pragma GCC reset_options
#pragma GCC target("stack-size=2048,stack-guard=0")
void p0(int x)
{
bar(x);
bar(x + 1);
}
#pragma GCC reset_options
__attribute__ ((target("stack-size=4096,stack-guard=0")))
void a1(int x)
{
bar(x);
bar(x + 1);
}
__attribute__ ((target("stack-size=8192,stack-guard=0")))
void a0(int x)
{
bar(x);
bar(x + 1);
}
void d(int x)
{
bar(x);
bar(x + 1);
}
