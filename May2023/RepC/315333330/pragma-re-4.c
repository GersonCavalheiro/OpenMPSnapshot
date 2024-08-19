#define foo     _foo
#define foo64   _foo64
extern int foo(void);
extern int foo64(void);
#pragma redefine_extname foo64 foo
int
bar()
{
return (foo64());
}
