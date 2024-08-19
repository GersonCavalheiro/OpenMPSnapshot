#include <stdio.h>
#pragma GCC target("no-backchain")
void p0(void)
{
printf ((void *)0);
}
#pragma GCC reset_options
__attribute__ ((target("no-backchain")))
void a0(void)
{
printf ((void *)0);
}
