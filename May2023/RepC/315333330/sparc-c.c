#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "c-family/c-common.h"
#include "c-family/c-pragma.h"
void
sparc_target_macros (void)
{
builtin_define_std ("sparc");
if (TARGET_ARCH64)
{
cpp_assert (parse_in, "cpu=sparc64");
cpp_assert (parse_in, "machine=sparc64");
}
else
{
cpp_assert (parse_in, "cpu=sparc");
cpp_assert (parse_in, "machine=sparc");
}
if (TARGET_VIS4B)
{
cpp_define (parse_in, "__VIS__=0x410");
cpp_define (parse_in, "__VIS=0x410");
}
else if (TARGET_VIS4)
{
cpp_define (parse_in, "__VIS__=0x400");
cpp_define (parse_in, "__VIS=0x400");
}
else if (TARGET_VIS3)
{
cpp_define (parse_in, "__VIS__=0x300");
cpp_define (parse_in, "__VIS=0x300");
}
else if (TARGET_VIS2)
{
cpp_define (parse_in, "__VIS__=0x200");
cpp_define (parse_in, "__VIS=0x200");
}
else if (TARGET_VIS)
{
cpp_define (parse_in, "__VIS__=0x100");
cpp_define (parse_in, "__VIS=0x100");
}
}
