
#ifndef GLM_GTX_raw_data
#define GLM_GTX_raw_data GLM_VERSION

#include "../glm.hpp"
#include "../gtc/type_precision.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_raw_data extension included")
#endif

namespace glm
{

typedef uint8		byte;

typedef uint16		word;

typedef uint32		dword;

typedef uint64		qword;

}

#include "raw_data.inl"

#endif
