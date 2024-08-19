#pragma GCC push_options
#pragma GCC optimize ("-fno-stack-protector")
__attribute__((constructor)) void foo()
{
asm ("");
}
#pragma GCC pop_options
int main() { return 0; }
