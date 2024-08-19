#pragma GCC target("warn-dynamicstack")
void p1(int s)
{
char b[s];
} 
#pragma GCC reset_options
#pragma GCC target("no-warn-dynamicstack")
void p0(int s)
{
char b[s];
}
#pragma GCC reset_options
__attribute__ ((target("warn-dynamicstack")))
void a1(int s)
{
char b[s];
} 
__attribute__ ((target("no-warn-dynamicstack")))
void a0(int s)
{
char b[s];
}
void d(int s)
{
char b[s];
}
