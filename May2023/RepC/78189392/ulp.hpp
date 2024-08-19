#pragma once
#include "../detail/setup.hpp"
#include "../detail/precision.hpp"
#include "../detail/type_int.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_ulp extension included")
#endif
namespace glm
{
template <typename genType>
GLM_FUNC_DECL genType next_float(genType const & x);
template <typename genType>
GLM_FUNC_DECL genType prev_float(genType const & x);
template <typename genType>
GLM_FUNC_DECL genType next_float(genType const & x, uint const & Distance);
template <typename genType>
GLM_FUNC_DECL genType prev_float(genType const & x, uint const & Distance);
template <typename T>
GLM_FUNC_DECL uint float_distance(T const & x, T const & y);
template<typename T, template<typename> class vecType>
GLM_FUNC_DECL vecType<uint> float_distance(vecType<T> const & x, vecType<T> const & y);
}
#include "ulp.inl"
