
#ifndef GLM_GTX_simd_quat
#define GLM_GTX_simd_quat

#include "../glm.hpp"
#include "../gtc/quaternion.hpp"
#include "../gtx/fast_trigonometry.hpp"

#if(GLM_ARCH != GLM_ARCH_PURE)

#if(GLM_ARCH & GLM_ARCH_SSE2)
#   include "../gtx/simd_mat4.hpp"
#else
#	error "GLM: GLM_GTX_simd_quat requires compiler support of SSE2 through intrinsics"
#endif

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_simd_quat extension included")
#endif


#if (GLM_COMPILER & GLM_COMPILER_VC)
#   pragma warning(push)
#   pragma warning(disable:4201)   
#endif


namespace glm{
namespace detail
{
GLM_ALIGNED_STRUCT(16) fquatSIMD
{
enum ctor{null};
typedef __m128 value_type;
typedef std::size_t size_type;
static size_type value_size();

typedef fquatSIMD type;
typedef tquat<bool, defaultp> bool_type;

#ifdef GLM_SIMD_ENABLE_XYZW_UNION
union
{
__m128 Data;
struct {float x, y, z, w;};
};
#else
__m128 Data;
#endif


fquatSIMD();
fquatSIMD(__m128 const & Data);
fquatSIMD(fquatSIMD const & q);


explicit fquatSIMD(
ctor);
explicit fquatSIMD(
float const & w, 
float const & x, 
float const & y, 
float const & z);
explicit fquatSIMD(
quat const & v);
explicit fquatSIMD(
vec3 const & eulerAngles);



fquatSIMD& operator =(fquatSIMD const & q);
fquatSIMD& operator*=(float const & s);
fquatSIMD& operator/=(float const & s);
};



detail::fquatSIMD operator- (
detail::fquatSIMD const & q);

detail::fquatSIMD operator+ ( 
detail::fquatSIMD const & q, 
detail::fquatSIMD const & p); 

detail::fquatSIMD operator* ( 
detail::fquatSIMD const & q, 
detail::fquatSIMD const & p); 

detail::fvec4SIMD operator* (
detail::fquatSIMD const & q, 
detail::fvec4SIMD const & v);

detail::fvec4SIMD operator* (
detail::fvec4SIMD const & v,
detail::fquatSIMD const & q);

detail::fquatSIMD operator* (
detail::fquatSIMD const & q, 
float s);

detail::fquatSIMD operator* (
float s,
detail::fquatSIMD const & q);

detail::fquatSIMD operator/ (
detail::fquatSIMD const & q, 
float s);

}

typedef glm::detail::fquatSIMD simdQuat;


quat quat_cast(
detail::fquatSIMD const & x);

detail::fquatSIMD quatSIMD_cast(
detail::fmat4x4SIMD const & m);

template <typename T, precision P>
detail::fquatSIMD quatSIMD_cast(
detail::tmat4x4<T, P> const & m);

template <typename T, precision P>
detail::fquatSIMD quatSIMD_cast(
detail::tmat3x3<T, P> const & m);

detail::fmat4x4SIMD mat4SIMD_cast(
detail::fquatSIMD const & q);

mat4 mat4_cast(
detail::fquatSIMD const & q);


float length(
detail::fquatSIMD const & x);

detail::fquatSIMD normalize(
detail::fquatSIMD const & x);

float dot(
detail::fquatSIMD const & q1, 
detail::fquatSIMD const & q2);

detail::fquatSIMD mix(
detail::fquatSIMD const & x, 
detail::fquatSIMD const & y, 
float const & a);

detail::fquatSIMD lerp(
detail::fquatSIMD const & x, 
detail::fquatSIMD const & y, 
float const & a);

detail::fquatSIMD slerp(
detail::fquatSIMD const & x, 
detail::fquatSIMD const & y, 
float const & a);


detail::fquatSIMD fastMix(
detail::fquatSIMD const & x, 
detail::fquatSIMD const & y, 
float const & a);

detail::fquatSIMD fastSlerp(
detail::fquatSIMD const & x, 
detail::fquatSIMD const & y, 
float const & a);


detail::fquatSIMD conjugate(
detail::fquatSIMD const & q);

detail::fquatSIMD inverse(
detail::fquatSIMD const & q);

detail::fquatSIMD angleAxisSIMD(
float const & angle, 
vec3 const & axis);

detail::fquatSIMD angleAxisSIMD(
float const & angle, 
float const & x, 
float const & y, 
float const & z);


__m128 fastSin(__m128 x);


}

#include "simd_quat.inl"


#if (GLM_COMPILER & GLM_COMPILER_VC)
#   pragma warning(pop)
#endif


#endif

#endif
