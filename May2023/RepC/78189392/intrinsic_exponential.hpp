#pragma once
#include "setup.hpp"
#if(!(GLM_ARCH & GLM_ARCH_SSE2))
#	error "SSE2 instructions not supported or enabled"
#else
namespace glm{
namespace detail
{
}
}
#endif
