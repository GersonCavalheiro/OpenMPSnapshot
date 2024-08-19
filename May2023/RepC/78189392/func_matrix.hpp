#pragma once
#include "../detail/precision.hpp"
#include "../detail/setup.hpp"
#include "../detail/type_mat.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../mat2x2.hpp"
#include "../mat2x3.hpp"
#include "../mat2x4.hpp"
#include "../mat3x2.hpp"
#include "../mat3x3.hpp"
#include "../mat3x4.hpp"
#include "../mat4x2.hpp"
#include "../mat4x3.hpp"
#include "../mat4x4.hpp"
namespace glm{
namespace detail
{
template <typename T, precision P>
struct outerProduct_trait<T, P, tvec2, tvec2>
{
typedef tmat2x2<T, P> type;
};
template <typename T, precision P>
struct outerProduct_trait<T, P, tvec2, tvec3>
{
typedef tmat3x2<T, P> type;
};
template <typename T, precision P>
struct outerProduct_trait<T, P, tvec2, tvec4>
{
typedef tmat4x2<T, P> type;
};
template <typename T, precision P>
struct outerProduct_trait<T, P, tvec3, tvec2>
{
typedef tmat2x3<T, P> type;
};
template <typename T, precision P>
struct outerProduct_trait<T, P, tvec3, tvec3>
{
typedef tmat3x3<T, P> type;
};
template <typename T, precision P>
struct outerProduct_trait<T, P, tvec3, tvec4>
{
typedef tmat4x3<T, P> type;
};
template <typename T, precision P>
struct outerProduct_trait<T, P, tvec4, tvec2>
{
typedef tmat2x4<T, P> type;
};
template <typename T, precision P>
struct outerProduct_trait<T, P, tvec4, tvec3>
{
typedef tmat3x4<T, P> type;
};
template <typename T, precision P>
struct outerProduct_trait<T, P, tvec4, tvec4>
{
typedef tmat4x4<T, P> type;
};
}
template <typename T, precision P, template <typename, precision> class matType>
GLM_FUNC_DECL matType<T, P> matrixCompMult(matType<T, P> const & x, matType<T, P> const & y);
template <typename T, precision P, template <typename, precision> class vecTypeA, template <typename, precision> class vecTypeB>
GLM_FUNC_DECL typename detail::outerProduct_trait<T, P, vecTypeA, vecTypeB>::type outerProduct(vecTypeA<T, P> const & c, vecTypeB<T, P> const & r);
#	if((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC2012))
template <typename T, precision P, template <typename, precision> class matType>
GLM_FUNC_DECL typename matType<T, P>::transpose_type transpose(matType<T, P> const & x);
#	endif
template <typename T, precision P, template <typename, precision> class matType>
GLM_FUNC_DECL T determinant(matType<T, P> const & m);
template <typename T, precision P, template <typename, precision> class matType>
GLM_FUNC_DECL matType<T, P> inverse(matType<T, P> const & m);
}
#include "func_matrix.inl"
