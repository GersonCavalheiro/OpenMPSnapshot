
#ifndef GLM_GTX_dual_quaternion
#define GLM_GTX_dual_quaternion

#include "../glm.hpp"
#include "../gtc/constants.hpp"
#include "../gtc/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_dual_quaternion extension included")
#endif

namespace glm{
namespace detail
{
template <typename T, precision P>
struct tdualquat
{
enum ctor{null};

typedef glm::detail::tquat<T, P> part_type;

public:
glm::detail::tquat<T, P> real, dual;

GLM_FUNC_DECL GLM_CONSTEXPR int length() const;

GLM_FUNC_DECL tdualquat();
GLM_FUNC_DECL explicit tdualquat(tquat<T, P> const & real);
GLM_FUNC_DECL tdualquat(tquat<T, P> const & real,tquat<T, P> const & dual);
GLM_FUNC_DECL tdualquat(tquat<T, P> const & orientation,tvec3<T, P> const& translation);

GLM_FUNC_DECL explicit tdualquat(tmat2x4<T, P> const & holder_mat);
GLM_FUNC_DECL explicit tdualquat(tmat3x4<T, P> const & aug_mat);

GLM_FUNC_DECL part_type & operator[](int i);
GLM_FUNC_DECL part_type const & operator[](int i) const;

GLM_FUNC_DECL tdualquat<T, P> & operator*=(T const & s);
GLM_FUNC_DECL tdualquat<T, P> & operator/=(T const & s);
};

template <typename T, precision P>
GLM_FUNC_DECL detail::tquat<T, P> operator- (
detail::tquat<T, P> const & q);

template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> operator+ (
detail::tdualquat<T, P> const & q,
detail::tdualquat<T, P> const & p);

template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> operator* (
detail::tdualquat<T, P> const & q,
detail::tdualquat<T, P> const & p);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> operator* (
detail::tquat<T, P> const & q,
detail::tvec3<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec3<T, P> operator* (
detail::tvec3<T, P> const & v,
detail::tquat<T, P> const & q);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec4<T, P> operator* (
detail::tquat<T, P> const & q,
detail::tvec4<T, P> const & v);

template <typename T, precision P>
GLM_FUNC_DECL detail::tvec4<T, P> operator* (
detail::tvec4<T, P> const & v,
detail::tquat<T, P> const & q);

template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> operator* (
detail::tdualquat<T, P> const & q,
T const & s);

template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> operator* (
T const & s,
detail::tdualquat<T, P> const & q);

template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> operator/ (
detail::tdualquat<T, P> const & q,
T const & s);
} 


template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> normalize(
detail::tdualquat<T, P> const & q);

template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> lerp(
detail::tdualquat<T, P> const & x,
detail::tdualquat<T, P> const & y,
T const & a);

template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> inverse(
detail::tdualquat<T, P> const & q);



template <typename T, precision P>
GLM_FUNC_DECL detail::tmat2x4<T, P> mat2x4_cast(
detail::tdualquat<T, P> const & x);

template <typename T, precision P>
GLM_FUNC_DECL detail::tmat3x4<T, P> mat3x4_cast(
detail::tdualquat<T, P> const & x);

template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> dualquat_cast(
detail::tmat2x4<T, P> const & x);

template <typename T, precision P>
GLM_FUNC_DECL detail::tdualquat<T, P> dualquat_cast(
detail::tmat3x4<T, P> const & x);


typedef detail::tdualquat<float, lowp>		lowp_dualquat;

typedef detail::tdualquat<float, mediump>	mediump_dualquat;

typedef detail::tdualquat<float, highp>		highp_dualquat;


typedef detail::tdualquat<float, lowp>		lowp_fdualquat;

typedef detail::tdualquat<float, mediump>	mediump_fdualquat;

typedef detail::tdualquat<float, highp>		highp_fdualquat;


typedef detail::tdualquat<double, lowp>		lowp_ddualquat;

typedef detail::tdualquat<double, mediump>	mediump_ddualquat;

typedef detail::tdualquat<double, highp>	highp_ddualquat;


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

#endif
