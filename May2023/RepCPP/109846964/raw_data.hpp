
#ifndef GLM_GTX_raw_data
#define GLM_GTX_raw_data

#include "../detail/setup.hpp"
#include "../detail/type_int.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_raw_data extension included")
#endif

namespace glm
{

typedef detail::uint8		byte;

typedef detail::uint16		word;

typedef detail::uint32		dword;

typedef detail::uint64		qword;

}

#include "raw_data.inl"

#endif
