void foo(void) { }
#pragma GCC target("arch=z196")
__attribute__ ((target("arch=z10")))
void bar(void) { }
