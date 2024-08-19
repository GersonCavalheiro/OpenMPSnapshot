#include "arm_acle.h"
__attribute__ ((target ("+crc+nocrypto")))
int
foo (uint32_t a, uint8_t b)
{
return __crc32b (a, b);
}
