#include <stdio.h>
#pragma GCC target("backchain")
void p1(void)
{
printf ((void *)0);
}
#pragma GCC reset_options
__attribute__ ((target("backchain")))
void a1(void)
{
printf ((void *)0);
}
