#ifndef __PRAGMA_REDEFINE_EXTNAME
#error 
#endif
#pragma redefine_extname foo bar
extern "C" int foo(void);
int (*p)(void) = foo;
#pragma redefine_extname baz baq
extern int baz(void);
int (*q)(void) = baz;
#pragma redefine_extname gee tut
namespace somewhere {
extern "C" int gee(void);
int (*r)(void) = gee;
extern "C" int whiz(void);
int whiz(int);
}
#pragma redefine_extname whiz bang
int (*s)() = somewhere::whiz;
namespace elsewhere {
extern "C" int whiz(void);
}
int (*t)() = elsewhere::whiz;
namespace A
{
extern "C" int boo(void);
}
namespace B
{
extern "C" int boo(void);
}
#pragma redefine_extname boo eek
int (*u)() = A::boo;
int (*v)() = B::boo;
