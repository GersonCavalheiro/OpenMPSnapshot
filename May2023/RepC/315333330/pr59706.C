extern struct S s;
struct T { T (); ~T (); int t; } t;
void
foo ()
{
#pragma GCC ivdep
while (s)	
;
}
void
bar ()
{
#pragma GCC ivdep
while (t)	
;
}
