#pragma once
#include "../glm.hpp"
#include "../gtc/constants.hpp"
#include "../gtc/quaternion.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTX_dual_quaternion extension included")
#endif
namespace glm
{
template <typename T, precision P = defaultp>
struct tdualquat
{
typedef T value_type;
typedef glm::tquat<T, P> part_type;
#		ifdef GLM_META_PROG_HELPERS
static GLM_RELAXED_CONSTEXPR length_t components = 2;
static GLM_RELAXED_CONSTEXPR precision prec = P;
#		endif
glm::tquat<T, P> real, dual;
#		ifdef GLM_FORCE_SIZE_FUNC
typedef size_t size_type;
GLM_FUNC_DECL GLM_CONSTEXPR size_type size() const;
GLM_FUNC_DECL part_type & operator[](size_type i);
GLM_FUNC_DECL part_type const & operator[](size_type i) const;
#		else
typedef length_t length_type;
GLM_FUNC_DECL GLM_CONSTEXPR length_type length() const;
GLM_FUNC_DECL part_type & operator[](length_type i);
GLM_FUNC_DECL part_type const & operator[](length_type i) const;
#		endif
GLM_FUNC_DECL tdualquat() GLM_DEFAULT_CTOR;
GLM_FUNC_DECL tdualquat(tdualquat<T, P> const & d) GLM_DEFAULT;
template <precision Q>
GLM_FUNC_DECL tdualquat(tdualquat<T, Q> const & d);
GLM_FUNC_DECL explicit tdualquat(ctor);
GLM_FUNC_DECL explicit tdualquat(tquat<T, P> const & real);
GLM_FUNC_DECL tdualquat(tquat<T, P> const & orientation, tvec3<T, P> const & translation);
GLM_FUNC_DECL tdualquat(tquat<T, P> const & real, tquat<T, P> const & dual);
template <typename U, precision Q>
GLM_FUNC_DECL GLM_EXPLICIT tdualquat(tdualquat<U, Q> const & q);
GLM_FUNC_DECL explicit tdualquat(tmat2x4<T, P> const & holder_mat);
GLM_FUNC_DECL explicit tdualquat(tmat3x4<T, P> const & aug_mat);
GLM_FUNC_DECL tdualquat<T, P> & operator=(tdualquat<T, P> const & m) GLM_DEFAULT;
template <typename U>
GLM_FUNC_DECL tdualquat<T, P> & operator=(tdualquat<U, P> const & m);
template <typename U>
GLM_FUNC_DECL tdualquat<T, P> & operator*=(U s);
template <typename U>
GLM_FUNC_DECL tdualquat<T, P> & operator/=(U s);
};
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> operator+(tdualquat<T, P> const & q);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> operator-(tdualquat<T, P> const & q);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> operator+(tdualquat<T, P> const & q, tdualquat<T, P> const & p);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> operator*(tdualquat<T, P> const & q, tdualquat<T, P> const & p);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> operator*(tdualquat<T, P> const & q, tvec3<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec3<T, P> operator*(tvec3<T, P> const & v, tdualquat<T, P> const & q);
template <typename T, precision P>
GLM_FUNC_DECL tvec4<T, P> operator*(tdualquat<T, P> const & q, tvec4<T, P> const & v);
template <typename T, precision P>
GLM_FUNC_DECL tvec4<T, P> operator*(tvec4<T, P> const & v, tdualquat<T, P> const & q);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> operator*(tdualquat<T, P> const & q, T const & s);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> operator*(T const & s, tdualquat<T, P> const & q);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> operator/(tdualquat<T, P> const & q, T const & s);
template <typename T, precision P>
GLM_FUNC_DECL bool operator==(tdualquat<T, P> const & q1, tdualquat<T, P> const & q2);
template <typename T, precision P>
GLM_FUNC_DECL bool operator!=(tdualquat<T, P> const & q1, tdualquat<T, P> const & q2);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> normalize(tdualquat<T, P> const & q);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> lerp(tdualquat<T, P> const & x, tdualquat<T, P> const & y, T const & a);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> inverse(tdualquat<T, P> const & q);
template <typename T, precision P>
GLM_FUNC_DECL tmat2x4<T, P> mat2x4_cast(tdualquat<T, P> const & x);
template <typename T, precision P>
GLM_FUNC_DECL tmat3x4<T, P> mat3x4_cast(tdualquat<T, P> const & x);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> dualquat_cast(tmat2x4<T, P> const & x);
template <typename T, precision P>
GLM_FUNC_DECL tdualquat<T, P> dualquat_cast(tmat3x4<T, P> const & x);
typedef tdualquat<float, lowp>		lowp_dualquat;
typedef tdualquat<float, mediump>	mediump_dualquat;
typedef tdualquat<float, highp>		highp_dualquat;
typedef tdualquat<float, lowp>		lowp_fdualquat;
typedef tdualquat<float, mediump>	mediump_fdualquat;
typedef tdualquat<float, highp>		highp_fdualquat;
typedef tdualquat<double, lowp>		lowp_ddualquat;
typedef tdualquat<double, mediump>	mediump_ddualquat;
typedef tdualquat<double, highp>	highp_ddualquat;
#if(!defined(GLM_PRECISION_HIGHP_FLOAT) && !defined(GLM_PRECISION_MEDIUMP_FLOAT) && !defined(GLM_PRECISION_LOWP_FLOAT))
typedef highp_fdualquat			dualquat;
typedef highp_fdualquat			fdualquat;
#elif(defined(GLM_PRECISION_HIGHP_FLOAT) && !defined(GLM_PRECISION_MEDIUMP_FLOAT) && !defined(GLM_PRECISION_LOWP_FLOAT))
typedef highp_fdualquat			dualquat;
typedef highp_fdualquat			fdualquat;
#elif(!defined(GLM_PRECISION_HIGHP_FLOAT) && defined(GLM_PRECISION_MEDIUMP_FLOAT) && !defined(GLM_PRECISION_LOWP_FLOAT))
typedef mediump_fdualquat		dualquat;
typedef mediump_fdualquat		fdualquat;
#elif(!defined(GLM_PRECISION_HIGHP_FLOAT) && !defined(GLM_PRECISION_MEDIUMP_FLOAT) && defined(GLM_PRECISION_LOWP_FLOAT))
typedef lowp_fdualquat			dualquat;
typedef lowp_fdualquat			fdualquat;
#else
#	error "GLM error: multiple default precision requested for single-precision floating-point types"
#endif
#if(!defined(GLM_PRECISION_HIGHP_DOUBLE) && !defined(GLM_PRECISION_MEDIUMP_DOUBLE) && !defined(GLM_PRECISION_LOWP_DOUBLE))
typedef highp_ddualquat			ddualquat;
#elif(defined(GLM_PRECISION_HIGHP_DOUBLE) && !defined(GLM_PRECISION_MEDIUMP_DOUBLE) && !defined(GLM_PRECISION_LOWP_DOUBLE))
typedef highp_ddualquat			ddualquat;
#elif(!defined(GLM_PRECISION_HIGHP_DOUBLE) && defined(GLM_PRECISION_MEDIUMP_DOUBLE) && !defined(GLM_PRECISION_LOWP_DOUBLE))
typedef mediump_ddualquat		ddualquat;
#elif(!defined(GLM_PRECISION_HIGHP_DOUBLE) && !defined(GLM_PRECISION_MEDIUMP_DOUBLE) && defined(GLM_PRECISION_LOWP_DOUBLE))
typedef lowp_ddualquat			ddualquat;
#else
#	error "GLM error: Multiple default precision requested for double-precision floating-point types"
#endif
} 
#include "dual_quaternion.inl"
