static int foo()
{
return 0;
}
#pragma GCC push_options
#pragma GCC optimize ("-fno-stack-protector")
int main() { foo (); }
#pragma GCC pop_options
