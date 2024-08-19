
#ifndef GLM_GTC_matrix_integer
#define GLM_GTC_matrix_integer GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTC_matrix_integer extension included")
#endif

namespace glm
{

typedef detail::tmat2x2<highp_int>				highp_imat2;	

typedef detail::tmat3x3<highp_int>				highp_imat3;

typedef detail::tmat4x4<highp_int>				highp_imat4;

typedef detail::tmat2x2<highp_int>				highp_imat2x2;

typedef detail::tmat2x3<highp_int>				highp_imat2x3;

typedef detail::tmat2x4<highp_int>				highp_imat2x4;

typedef detail::tmat3x2<highp_int>				highp_imat3x2;

typedef detail::tmat3x3<highp_int>				highp_imat3x3;

typedef detail::tmat3x4<highp_int>				highp_imat3x4;

typedef detail::tmat4x2<highp_int>				highp_imat4x2;

typedef detail::tmat4x3<highp_int>				highp_imat4x3;

typedef detail::tmat4x4<highp_int>				highp_imat4x4;


typedef detail::tmat2x2<mediump_int>			mediump_imat2;

typedef detail::tmat3x3<mediump_int>			mediump_imat3;

typedef detail::tmat4x4<mediump_int>			mediump_imat4;


typedef detail::tmat2x2<mediump_int>			mediump_imat2x2;

typedef detail::tmat2x3<mediump_int>			mediump_imat2x3;

typedef detail::tmat2x4<mediump_int>			mediump_imat2x4;

typedef detail::tmat3x2<mediump_int>			mediump_imat3x2;

typedef detail::tmat3x3<mediump_int>			mediump_imat3x3;

typedef detail::tmat3x4<mediump_int>			mediump_imat3x4;

typedef detail::tmat4x2<mediump_int>			mediump_imat4x2;

typedef detail::tmat4x3<mediump_int>			mediump_imat4x3;

typedef detail::tmat4x4<mediump_int>			mediump_imat4x4;


typedef detail::tmat2x2<lowp_int>				lowp_imat2;

typedef detail::tmat3x3<lowp_int>				lowp_imat3;

typedef detail::tmat4x4<lowp_int>				lowp_imat4;


typedef detail::tmat2x2<lowp_int>				lowp_imat2x2;

typedef detail::tmat2x3<lowp_int>				lowp_imat2x3;

typedef detail::tmat2x4<lowp_int>				lowp_imat2x4;

typedef detail::tmat3x2<lowp_int>				lowp_imat3x2;

typedef detail::tmat3x3<lowp_int>				lowp_imat3x3;

typedef detail::tmat3x4<lowp_int>				lowp_imat3x4;

typedef detail::tmat4x2<lowp_int>				lowp_imat4x2;

typedef detail::tmat4x3<lowp_int>				lowp_imat4x3;

typedef detail::tmat4x4<lowp_int>				lowp_imat4x4;


typedef detail::tmat2x2<highp_uint>				highp_umat2;	

typedef detail::tmat3x3<highp_uint>				highp_umat3;

typedef detail::tmat4x4<highp_uint>				highp_umat4;

typedef detail::tmat2x2<highp_uint>				highp_umat2x2;

typedef detail::tmat2x3<highp_uint>				highp_umat2x3;

typedef detail::tmat2x4<highp_uint>				highp_umat2x4;

typedef detail::tmat3x2<highp_uint>				highp_umat3x2;

typedef detail::tmat3x3<highp_uint>				highp_umat3x3;

typedef detail::tmat3x4<highp_uint>				highp_umat3x4;

typedef detail::tmat4x2<highp_uint>				highp_umat4x2;

typedef detail::tmat4x3<highp_uint>				highp_umat4x3;

typedef detail::tmat4x4<highp_uint>				highp_umat4x4;


typedef detail::tmat2x2<mediump_uint>			mediump_umat2;

typedef detail::tmat3x3<mediump_uint>			mediump_umat3;

typedef detail::tmat4x4<mediump_uint>			mediump_umat4;


typedef detail::tmat2x2<mediump_uint>			mediump_umat2x2;

typedef detail::tmat2x3<mediump_uint>			mediump_umat2x3;

typedef detail::tmat2x4<mediump_uint>			mediump_umat2x4;

typedef detail::tmat3x2<mediump_uint>			mediump_umat3x2;

typedef detail::tmat3x3<mediump_uint>			mediump_umat3x3;

typedef detail::tmat3x4<mediump_uint>			mediump_umat3x4;

typedef detail::tmat4x2<mediump_uint>			mediump_umat4x2;

typedef detail::tmat4x3<mediump_uint>			mediump_umat4x3;

typedef detail::tmat4x4<mediump_uint>			mediump_umat4x4;


typedef detail::tmat2x2<lowp_uint>				lowp_umat2;

typedef detail::tmat3x3<lowp_uint>				lowp_umat3;

typedef detail::tmat4x4<lowp_uint>				lowp_umat4;


typedef detail::tmat2x2<lowp_uint>				lowp_umat2x2;

typedef detail::tmat2x3<lowp_uint>				lowp_umat2x3;

typedef detail::tmat2x4<lowp_uint>				lowp_umat2x4;

typedef detail::tmat3x2<lowp_uint>				lowp_umat3x2;

typedef detail::tmat3x3<lowp_uint>				lowp_umat3x3;

typedef detail::tmat3x4<lowp_uint>				lowp_umat3x4;

typedef detail::tmat4x2<lowp_uint>				lowp_umat4x2;

typedef detail::tmat4x3<lowp_uint>				lowp_umat4x3;

typedef detail::tmat4x4<lowp_uint>				lowp_umat4x4;

#if(defined(GLM_PRECISION_HIGHP_INT))
typedef highp_imat2								imat2;
typedef highp_imat3								imat3;
typedef highp_imat4								imat4;
typedef highp_imat2x2							imat2x2;
typedef highp_imat2x3							imat2x3;
typedef highp_imat2x4							imat2x4;
typedef highp_imat3x2							imat3x2;
typedef highp_imat3x3							imat3x3;
typedef highp_imat3x4							imat3x4;
typedef highp_imat4x2							imat4x2;
typedef highp_imat4x3							imat4x3;
typedef highp_imat4x4							imat4x4;
#elif(defined(GLM_PRECISION_LOWP_INT))
typedef lowp_imat2								imat2;
typedef lowp_imat3								imat3;
typedef lowp_imat4								imat4;
typedef lowp_imat2x2							imat2x2;
typedef lowp_imat2x3							imat2x3;
typedef lowp_imat2x4							imat2x4;
typedef lowp_imat3x2							imat3x2;
typedef lowp_imat3x3							imat3x3;
typedef lowp_imat3x4							imat3x4;
typedef lowp_imat4x2							imat4x2;
typedef lowp_imat4x3							imat4x3;
typedef lowp_imat4x4							imat4x4;
#else 

typedef mediump_imat2							imat2;

typedef mediump_imat3							imat3;

typedef mediump_imat4							imat4;

typedef mediump_imat2x2							imat2x2;

typedef mediump_imat2x3							imat2x3;

typedef mediump_imat2x4							imat2x4;

typedef mediump_imat3x2							imat3x2;

typedef mediump_imat3x3							imat3x3;

typedef mediump_imat3x4							imat3x4;

typedef mediump_imat4x2							imat4x2;

typedef mediump_imat4x3							imat4x3;

typedef mediump_imat4x4							imat4x4;
#endif

#if(defined(GLM_PRECISION_HIGHP_UINT))
typedef highp_umat2								umat2;
typedef highp_umat3								umat3;
typedef highp_umat4								umat4;
typedef highp_umat2x2							umat2x2;
typedef highp_umat2x3							umat2x3;
typedef highp_umat2x4							umat2x4;
typedef highp_umat3x2							umat3x2;
typedef highp_umat3x3							umat3x3;
typedef highp_umat3x4							umat3x4;
typedef highp_umat4x2							umat4x2;
typedef highp_umat4x3							umat4x3;
typedef highp_umat4x4							umat4x4;
#elif(defined(GLM_PRECISION_LOWP_UINT))
typedef lowp_umat2								umat2;
typedef lowp_umat3								umat3;
typedef lowp_umat4								umat4;
typedef lowp_umat2x2							umat2x2;
typedef lowp_umat2x3							umat2x3;
typedef lowp_umat2x4							umat2x4;
typedef lowp_umat3x2							umat3x2;
typedef lowp_umat3x3							umat3x3;
typedef lowp_umat3x4							umat3x4;
typedef lowp_umat4x2							umat4x2;
typedef lowp_umat4x3							umat4x3;
typedef lowp_umat4x4							umat4x4;
#else 

typedef mediump_umat2							umat2;

typedef mediump_umat3							umat3;

typedef mediump_umat4							umat4;

typedef mediump_umat2x2							umat2x2;

typedef mediump_umat2x3							umat2x3;

typedef mediump_umat2x4							umat2x4;

typedef mediump_umat3x2							umat3x2;

typedef mediump_umat3x3							umat3x3;

typedef mediump_umat3x4							umat3x4;

typedef mediump_umat4x2							umat4x2;

typedef mediump_umat4x3							umat4x3;

typedef mediump_umat4x4							umat4x4;
#endif

}

#endif
