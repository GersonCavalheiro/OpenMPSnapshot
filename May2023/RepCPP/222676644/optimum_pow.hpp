
#ifndef GLM_GTX_optimum_pow
#define GLM_GTX_optimum_pow GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_optimum_pow extension included")
#endif

namespace glm{
namespace gtx
{

template <typename genType> 
genType pow2(const genType& x);

template <typename genType> 
genType pow3(const genType& x);

template <typename genType> 
genType pow4(const genType& x);

bool powOfTwo(int num);

detail::tvec2<bool> powOfTwo(const detail::tvec2<int>& x);

detail::tvec3<bool> powOfTwo(const detail::tvec3<int>& x);

detail::tvec4<bool> powOfTwo(const detail::tvec4<int>& x);

}
}

#include "optimum_pow.inl"

#endif
