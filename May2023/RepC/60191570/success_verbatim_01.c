#include <stdio.h>
#pragma mcc verbatim start
static void f(void)
{
printf("Hello world\n");
}
#pragma mcc verbatim end
static void f(void);
int main(int argc, char *argv[])
{
f();
return 0;
}
