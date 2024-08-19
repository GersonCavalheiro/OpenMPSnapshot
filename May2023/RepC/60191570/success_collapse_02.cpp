#include <iostream>
#include <cassert>
int main()
{
char lb_i = 0;
char lb_j = 0;
char lb_k = 0;
char ub_i = 4;
char ub_j = 4;
char ub_k = 8;
char s_i = 1;
char s_j = 1;
char s_k = 1;
unsigned int result = 0;
#pragma hlt collapse(3)
for (char i = lb_i; i < ub_i; i += s_i)
for (char j = lb_j; j < ub_j; j += s_j)
for (char k = lb_k; k < ub_k; k += s_k)
{
result++;
}
assert(result == (1 << 7));
std::cout << "result: " << result << std::endl;
}
