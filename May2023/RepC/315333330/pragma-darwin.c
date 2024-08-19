extern void abort(void);
#pragma mark hey hey ho
#pragma options 23  
#pragma options align  
#pragma options align natural 
#pragma options align=45 
#pragma options align=foo 
#ifndef __LP64__
#pragma options align=mac68k
struct s1 { short f1; int f2; };
#endif
#pragma options align=power
struct s2 { short f1; int f2; };
#ifndef __LP64__
#pragma options align=mac68k
struct s3 { short f1; int f2; };
#endif
#pragma options align=reset
struct s4 { short f1; int f2; };
#pragma options align=natural foo 
#pragma segment foo
int
main ()
{
int x, z;  
#pragma unused (x, y)
#ifndef __LP64__
if (sizeof (struct s1) != 6)
abort ();
#endif
if (sizeof (struct s2) != 8)
abort ();
#ifndef __LP64__
if (sizeof (struct s3) != 6)
abort ();
#endif
if (sizeof (struct s4) != 8)
abort ();
return 0;
}
void
unused_err_test ()
{
int a, b;
#pragma unused  
#pragma unused (a  
#pragma unused (b) foo 
}
