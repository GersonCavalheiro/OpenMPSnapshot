#include <stdint.h>
extern uint32_t bar();
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+crc")
uint32_t crc32cw(uint32_t crc, uint32_t val)
{
uint32_t res;
asm("crc32cw %0, %1, %2" : "=r"(res) : "r"(crc), "r"(val));
return res;
}
#pragma GCC pop_options
uint32_t restored ()
{
return bar();
}
