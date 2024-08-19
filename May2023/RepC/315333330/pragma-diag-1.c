void foo (int);
int
main()
{
int a;
int b;
int c;
int d;
#pragma GCC diagnostic error "-Wuninitialized"
foo(a);			
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
foo(b);
#pragma GCC diagnostic pop
foo(c);			
#pragma GCC diagnostic pop
foo(d);			
}
