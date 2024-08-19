#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "c-family/c-common.h"
void
rl78_register_pragmas (void)
{
c_register_addr_space ("__near", ADDR_SPACE_NEAR);
c_register_addr_space ("__far", ADDR_SPACE_FAR);
}
