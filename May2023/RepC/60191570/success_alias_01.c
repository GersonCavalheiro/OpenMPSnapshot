#include <stdlib.h>
extern int f_(int x) __attribute__((alias("f")));
int f(int x)
{
return x + 1;
}
int main(int argc, char* argv[])
{
int x = 41;
#pragma mcc verbatim start
x = f_(x);
#pragma mcc verbatim end
if (x != 42)
abort();
}
