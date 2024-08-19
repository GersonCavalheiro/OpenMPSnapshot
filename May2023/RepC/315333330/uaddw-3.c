#pragma GCC target "+nosve"
int 
t6(int len, void * dummy, char * __restrict x)
{
len = len & ~31;
unsigned short result = 0;
__asm volatile ("");
for (int i = 0; i < len; i++)
result += x[i];
return result;
}
