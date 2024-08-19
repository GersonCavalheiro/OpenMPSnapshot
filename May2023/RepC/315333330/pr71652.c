#pragma GCC push_options
#pragma GCC target ("arch=generic") 
__attribute__((constructor)) void foo()
{
asm ("");
}
#pragma GCC pop_options
int main() { return 0; }
