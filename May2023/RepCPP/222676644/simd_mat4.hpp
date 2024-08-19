
#ifndef GLM_GTX_simd_mat4
#define GLM_GTX_simd_mat4 GLM_VERSION

#include "../glm.hpp"

#if(GLM_ARCH != GLM_ARCH_PURE)

#if(GLM_ARCH & GLM_ARCH_SSE2)
#	include "../core/intrinsic_matrix.hpp"
#	include "../gtx/simd_vec4.hpp"
#else
#	error "GLM: GLM_GTX_simd_mat4 requires compiler support of SSE2 through intrinsics"
#endif

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_simd_mat4 extension included")
#endif

namespace glm{
namespace detail
{
GLM_ALIGNED_STRUCT(16) fmat4x4SIMD
{
enum ctor{null};

typedef float value_type;
typedef fvec4SIMD col_type;
typedef fvec4SIMD row_type;
typedef std::size_t size_type;
static size_type value_size();
static size_type col_size();
static size_type row_size();
static bool is_matrix();

fvec4SIMD Data[4];


fmat4x4SIMD();
explicit fmat4x4SIMD(float const & s);
explicit fmat4x4SIMD(
float const & x0, float const & y0, float const & z0, float const & w0,
float const & x1, float const & y1, float const & z1, float const & w1,
float const & x2, float const & y2, float const & z2, float const & w2,
float const & x3, float const & y3, float const & z3, float const & w3);
explicit fmat4x4SIMD(
fvec4SIMD const & v0,
fvec4SIMD const & v1,
fvec4SIMD const & v2,
fvec4SIMD const & v3);
explicit fmat4x4SIMD(
tmat4x4<float> const & m);
explicit fmat4x4SIMD(
__m128 const in[4]);



fvec4SIMD & operator[](size_type i);
fvec4SIMD const & operator[](size_type i) const;

fmat4x4SIMD & operator= (fmat4x4SIMD const & m);
fmat4x4SIMD & operator+= (float const & s);
fmat4x4SIMD & operator+= (fmat4x4SIMD const & m);
fmat4x4SIMD & operator-= (float const & s);
fmat4x4SIMD & operator-= (fmat4x4SIMD const & m);
fmat4x4SIMD & operator*= (float const & s);
fmat4x4SIMD & operator*= (fmat4x4SIMD const & m);
fmat4x4SIMD & operator/= (float const & s);
fmat4x4SIMD & operator/= (fmat4x4SIMD const & m);
fmat4x4SIMD & operator++ ();
fmat4x4SIMD & operator-- ();
};

fmat4x4SIMD operator+ (fmat4x4SIMD const & m, float const & s);
fmat4x4SIMD operator+ (float const & s, fmat4x4SIMD const & m);
fmat4x4SIMD operator+ (fmat4x4SIMD const & m1, fmat4x4SIMD const & m2);

fmat4x4SIMD operator- (fmat4x4SIMD const & m, float const & s);
fmat4x4SIMD operator- (float const & s, fmat4x4SIMD const & m);
fmat4x4SIMD operator- (fmat4x4SIMD const & m1, fmat4x4SIMD const & m2);

fmat4x4SIMD operator* (fmat4x4SIMD const & m, float const & s);
fmat4x4SIMD operator* (float const & s, fmat4x4SIMD const & m);

fvec4SIMD operator* (fmat4x4SIMD const & m, fvec4SIMD const & v);
fvec4SIMD operator* (fvec4SIMD const & v, fmat4x4SIMD const & m);

fmat4x4SIMD operator* (fmat4x4SIMD const & m1, fmat4x4SIMD const & m2);

fmat4x4SIMD operator/ (fmat4x4SIMD const & m, float const & s);
fmat4x4SIMD operator/ (float const & s, fmat4x4SIMD const & m);

fvec4SIMD operator/ (fmat4x4SIMD const & m, fvec4SIMD const & v);
fvec4SIMD operator/ (fvec4SIMD const & v, fmat4x4SIMD const & m);

fmat4x4SIMD operator/ (fmat4x4SIMD const & m1, fmat4x4SIMD const & m2);

fmat4x4SIMD const operator-  (fmat4x4SIMD const & m);
fmat4x4SIMD const operator-- (fmat4x4SIMD const & m, int);
fmat4x4SIMD const operator++ (fmat4x4SIMD const & m, int);
}

typedef detail::fmat4x4SIMD simdMat4;


detail::tmat4x4<float> mat4_cast(
detail::fmat4x4SIMD const & x);

detail::fmat4x4SIMD matrixCompMult(
detail::fmat4x4SIMD const & x,
detail::fmat4x4SIMD const & y);

detail::fmat4x4SIMD outerProduct(
detail::fvec4SIMD const & c,
detail::fvec4SIMD const & r);

detail::fmat4x4SIMD transpose(
detail::fmat4x4SIMD const & x);

float determinant(
detail::fmat4x4SIMD const & m);

detail::fmat4x4SIMD inverse(
detail::fmat4x4SIMD const & m);

}

#include "simd_mat4.inl"

#endif

#endif
