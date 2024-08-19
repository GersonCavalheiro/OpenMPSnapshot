#pragma GCC target ("sse2")
struct A
{ 
float V4SF __attribute__ ((vector_size (16)));
};
int
main (void)
{
struct A a = { { 0, 1, 2, 3 } };
foo (8.0L, a, 8.0L);
return 0;
}
