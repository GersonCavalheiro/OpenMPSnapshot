void f (unsigned);
#define CODE_WITH_WARNING \
int a; 	 \
f (a)	 
#pragma GCC diagnostic ignored "-Wuninitialized"
void
g (void)
{
CODE_WITH_WARNING;
}
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wuninitialized"
void
h (void)
{
CODE_WITH_WARNING; 
}
