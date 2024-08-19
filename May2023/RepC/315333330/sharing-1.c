int thrglobalvar;
#pragma omp threadprivate (thrglobalvar)
int globalvar;
const struct S
{
int x;
} constvar = { 8 };
struct T
{
static T t;
int i;
};
T T::t = { 6 };
const struct U
{
int x;
mutable int y;
} constmutvar = { 6, 4 };
int
foo (int x)
{
return x;
}
int
bar (int *x)
{
return *x;
}
int
baz (U u)
{
return u.x;
}
int
main (void)
{
static int thrlocvar;
#pragma omp threadprivate (thrlocvar)
static int locvar;
static int *p;
int i, j, s, l;
p = new int;
*p = 7;
s = 6;
l = 0;
#pragma omp parallel for  default (none) private (p) shared (s) 
for (i = 0; i < 64; i++)
{
int k = foo (0);	
k++;		
thrglobalvar++;	
thrlocvar++;	
foo (i);		
foo (constvar.x);	
foo (T::t.i);	
foo (*p);		
(*p)++;		
bar (p);		
foo (s);		
globalvar++;	
locvar++;		
l++;		
for (j = 0; j < 2; j++); 
baz (constmutvar);
}
return 0;
}
