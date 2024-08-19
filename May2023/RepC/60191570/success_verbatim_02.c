#include <stdio.h>
static void f(void)
{
#pragma mcc verbatim start
printf("Hello world\n");
#pragma mcc verbatim end
}
static void f(void);
int main(int argc, char *argv[])
{
f();
return 0;
}
