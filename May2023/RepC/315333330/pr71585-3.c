#pragma GCC push_options
#pragma GCC optimize ("-fno-stack-protector")
int foo()
{
return 0;
}
#pragma GCC pop_options
int main() { foo (); }
