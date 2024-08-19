
#ifndef GLM_GTX_vec1
#define GLM_GTX_vec1

#include "../glm.hpp"
#include "../detail/type_vec1.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_vec1 extension included")
#endif

namespace glm
{
typedef highp_vec1_t			highp_vec1;

typedef mediump_vec1_t			mediump_vec1;

typedef lowp_vec1_t				lowp_vec1;

typedef highp_ivec1_t			highp_ivec1;

typedef mediump_ivec1_t			mediump_ivec1;

typedef lowp_ivec1_t			lowp_ivec1;

typedef highp_uvec1_t			highp_uvec1;

typedef mediump_uvec1_t			mediump_uvec1;

typedef lowp_uvec1_t			lowp_uvec1;

typedef highp_bvec1_t			highp_bvec1;

typedef mediump_bvec1_t			mediump_bvec1;

typedef lowp_bvec1_t			lowp_bvec1;


#if(defined(GLM_PRECISION_HIGHP_BOOL))
typedef highp_bvec1				bvec1;
#elif(defined(GLM_PRECISION_MEDIUMP_BOOL))
typedef mediump_bvec1			bvec1;
#elif(defined(GLM_PRECISION_LOWP_BOOL))
typedef lowp_bvec1				bvec1;
#else
typedef highp_bvec1				bvec1;
#endif

#if(defined(GLM_PRECISION_HIGHP_FLOAT))
typedef highp_vec1				vec1;
#elif(defined(GLM_PRECISION_MEDIUMP_FLOAT))
typedef mediump_vec1			vec1;
#elif(defined(GLM_PRECISION_LOWP_FLOAT))
typedef lowp_vec1				vec1;
#else
typedef highp_vec1				vec1;
#endif

#if(defined(GLM_PRECISION_HIGHP_INT))
typedef highp_ivec1			ivec1;
#elif(defined(GLM_PRECISION_MEDIUMP_INT))
typedef mediump_ivec1		ivec1;
#elif(defined(GLM_PRECISION_LOWP_INT))
typedef lowp_ivec1			ivec1;
#else
typedef highp_ivec1			ivec1;
#endif

#if(defined(GLM_PRECISION_HIGHP_UINT))
typedef highp_uvec1			uvec1;
#elif(defined(GLM_PRECISION_MEDIUMP_UINT))
typedef mediump_uvec1		uvec1;
#elif(defined(GLM_PRECISION_LOWP_UINT))
typedef lowp_uvec1			uvec1;
#else
typedef highp_uvec1			uvec1;
#endif

}

#include "vec1.inl"

#endif

