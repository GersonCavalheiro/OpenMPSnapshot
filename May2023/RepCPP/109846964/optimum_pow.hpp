
#ifndef GLM_GTX_optimum_pow
#define GLM_GTX_optimum_pow

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_optimum_pow extension included")
#endif

namespace glm{
namespace gtx
{

template <typename genType>
GLM_FUNC_DECL genType pow2(const genType& x);

template <typename genType>
GLM_FUNC_DECL genType pow3(const genType& x);

template <typename genType>
GLM_FUNC_DECL genType pow4(const genType& x);

GLM_FUNC_DECL bool powOfTwo(int num);

template <precision P>
GLM_FUNC_DECL detail::tvec2<bool, P> powOfTwo(detail::tvec2<int, P> const & x);

template <precision P>
GLM_FUNC_DECL detail::tvec3<bool, P> powOfTwo(detail::tvec3<int, P> const & x);

template <precision P>
GLM_FUNC_DECL detail::tvec4<bool, P> powOfTwo(detail::tvec4<int, P> const & x);

}
}

#include "optimum_pow.inl"

#endif
