
#ifndef GLM_GTX_simd_vec4
#define GLM_GTX_simd_vec4 GLM_VERSION

#include "../glm.hpp"

#if(GLM_ARCH != GLM_ARCH_PURE)

#if(GLM_ARCH & GLM_ARCH_SSE2)
#	include "../core/intrinsic_common.hpp"
#	include "../core/intrinsic_geometric.hpp"
#else
#	error "GLM: GLM_GTX_simd_vec4 requires compiler support of SSE2 through intrinsics"
#endif

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_simd_vec4 extension included")
#endif


#if (GLM_COMPILER & GLM_COMPILER_VC)
#   pragma warning(push)
#   pragma warning(disable:4201)   
#endif


namespace glm{
namespace detail
{
GLM_ALIGNED_STRUCT(16) fvec4SIMD
{
enum ctor{null};
typedef __m128 value_type;
typedef std::size_t size_type;
static size_type value_size();

typedef fvec4SIMD type;
typedef tvec4<bool> bool_type;

#ifdef GLM_SIMD_ENABLE_XYZW_UNION
union
{
__m128 Data;
struct {float x, y, z, w;};
};
#else
__m128 Data;
#endif


fvec4SIMD();
fvec4SIMD(__m128 const & Data);
fvec4SIMD(fvec4SIMD const & v);


explicit fvec4SIMD(
ctor);
explicit fvec4SIMD(
float const & s);
explicit fvec4SIMD(
float const & x, 
float const & y, 
float const & z, 
float const & w);
explicit fvec4SIMD(
tvec4<float> const & v);


fvec4SIMD(vec2 const & v, float const & s1, float const & s2);
fvec4SIMD(float const & s1, vec2 const & v, float const & s2);
fvec4SIMD(float const & s1, float const & s2, vec2 const & v);
fvec4SIMD(vec3 const & v, float const & s);
fvec4SIMD(float const & s, vec3 const & v);
fvec4SIMD(vec2 const & v1, vec2 const & v2);


fvec4SIMD& operator= (fvec4SIMD const & v);
fvec4SIMD& operator+=(fvec4SIMD const & v);
fvec4SIMD& operator-=(fvec4SIMD const & v);
fvec4SIMD& operator*=(fvec4SIMD const & v);
fvec4SIMD& operator/=(fvec4SIMD const & v);

fvec4SIMD& operator+=(float const & s);
fvec4SIMD& operator-=(float const & s);
fvec4SIMD& operator*=(float const & s);
fvec4SIMD& operator/=(float const & s);

fvec4SIMD& operator++();
fvec4SIMD& operator--();


template <comp X, comp Y, comp Z, comp W>
fvec4SIMD& swizzle();
template <comp X, comp Y, comp Z, comp W>
fvec4SIMD swizzle() const;
template <comp X, comp Y, comp Z>
fvec4SIMD swizzle() const;
template <comp X, comp Y>
fvec4SIMD swizzle() const;
template <comp X>
fvec4SIMD swizzle() const;
};
}

typedef glm::detail::fvec4SIMD simdVec4;


detail::tvec4<float> vec4_cast(
detail::fvec4SIMD const & x);

detail::fvec4SIMD abs(detail::fvec4SIMD const & x);

detail::fvec4SIMD sign(detail::fvec4SIMD const & x);

detail::fvec4SIMD floor(detail::fvec4SIMD const & x);

detail::fvec4SIMD trunc(detail::fvec4SIMD const & x);

detail::fvec4SIMD round(detail::fvec4SIMD const & x);


detail::fvec4SIMD ceil(detail::fvec4SIMD const & x);

detail::fvec4SIMD fract(detail::fvec4SIMD const & x);

detail::fvec4SIMD mod(
detail::fvec4SIMD const & x, 
detail::fvec4SIMD const & y);

detail::fvec4SIMD mod(
detail::fvec4SIMD const & x, 
float const & y);


detail::fvec4SIMD min(
detail::fvec4SIMD const & x, 
detail::fvec4SIMD const & y);

detail::fvec4SIMD min(
detail::fvec4SIMD const & x, 
float const & y);

detail::fvec4SIMD max(
detail::fvec4SIMD const & x, 
detail::fvec4SIMD const & y);

detail::fvec4SIMD max(
detail::fvec4SIMD const & x, 
float const & y);

detail::fvec4SIMD clamp(
detail::fvec4SIMD const & x, 
detail::fvec4SIMD const & minVal, 
detail::fvec4SIMD const & maxVal); 

detail::fvec4SIMD clamp(
detail::fvec4SIMD const & x, 
float const & minVal, 
float const & maxVal); 

detail::fvec4SIMD mix(
detail::fvec4SIMD const & x, 
detail::fvec4SIMD const & y, 
detail::fvec4SIMD const & a);

detail::fvec4SIMD step(
detail::fvec4SIMD const & edge, 
detail::fvec4SIMD const & x);

detail::fvec4SIMD step(
float const & edge, 
detail::fvec4SIMD const & x);

detail::fvec4SIMD smoothstep(
detail::fvec4SIMD const & edge0, 
detail::fvec4SIMD const & edge1, 
detail::fvec4SIMD const & x);

detail::fvec4SIMD smoothstep(
float const & edge0, 
float const & edge1, 
detail::fvec4SIMD const & x);





detail::fvec4SIMD fma(
detail::fvec4SIMD const & a, 
detail::fvec4SIMD const & b, 
detail::fvec4SIMD const & c);



float length(
detail::fvec4SIMD const & x);

float fastLength(
detail::fvec4SIMD const & x);

float niceLength(
detail::fvec4SIMD const & x);

detail::fvec4SIMD length4(
detail::fvec4SIMD const & x);

detail::fvec4SIMD fastLength4(
detail::fvec4SIMD const & x);

detail::fvec4SIMD niceLength4(
detail::fvec4SIMD const & x);

float distance(
detail::fvec4SIMD const & p0,
detail::fvec4SIMD const & p1);

detail::fvec4SIMD distance4(
detail::fvec4SIMD const & p0,
detail::fvec4SIMD const & p1);

float simdDot(
detail::fvec4SIMD const & x,
detail::fvec4SIMD const & y);

detail::fvec4SIMD dot4(
detail::fvec4SIMD const & x,
detail::fvec4SIMD const & y);

detail::fvec4SIMD cross(
detail::fvec4SIMD const & x,
detail::fvec4SIMD const & y);

detail::fvec4SIMD normalize(
detail::fvec4SIMD const & x);

detail::fvec4SIMD fastNormalize(
detail::fvec4SIMD const & x);

detail::fvec4SIMD simdFaceforward(
detail::fvec4SIMD const & N,
detail::fvec4SIMD const & I,
detail::fvec4SIMD const & Nref);

detail::fvec4SIMD reflect(
detail::fvec4SIMD const & I,
detail::fvec4SIMD const & N);

detail::fvec4SIMD refract(
detail::fvec4SIMD const & I,
detail::fvec4SIMD const & N,
float const & eta);

detail::fvec4SIMD sqrt(
detail::fvec4SIMD const & x);

detail::fvec4SIMD niceSqrt(
detail::fvec4SIMD const & x);

detail::fvec4SIMD fastSqrt(
detail::fvec4SIMD const & x);

detail::fvec4SIMD inversesqrt(
detail::fvec4SIMD const & x);

detail::fvec4SIMD fastInversesqrt(
detail::fvec4SIMD const & x);

}

#include "simd_vec4.inl"


#if (GLM_COMPILER & GLM_COMPILER_VC)
#   pragma warning(pop)
#endif


#endif

#endif
