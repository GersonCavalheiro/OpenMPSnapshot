
#ifndef GLM_GTX_vec1
#define GLM_GTX_vec1 GLM_VERSION

#include "../glm.hpp"
#include "../core/type_vec1.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_vec1 extension included")
#endif

namespace glm
{
typedef detail::highp_vec1_t			highp_vec1;
typedef detail::mediump_vec1_t			mediump_vec1;
typedef detail::lowp_vec1_t				lowp_vec1;

typedef detail::highp_ivec1_t			highp_ivec1;
typedef detail::mediump_ivec1_t			mediump_ivec1;
typedef detail::lowp_ivec1_t			lowp_ivec1;

typedef detail::highp_uvec1_t			highp_uvec1;
typedef detail::mediump_uvec1_t			mediump_uvec1;
typedef detail::lowp_uvec1_t			lowp_uvec1;


typedef detail::tvec1<bool>	bvec1;

#if(defined(GLM_PRECISION_HIGHP_FLOAT))
typedef highp_vec1			vec1;
#elif(defined(GLM_PRECISION_MEDIUMP_FLOAT))
typedef mediump_vec1			vec1;
#elif(defined(GLM_PRECISION_LOWP_FLOAT))
typedef lowp_vec1			vec1;
#else
typedef mediump_vec1			vec1;
#endif

#if(defined(GLM_PRECISION_HIGHP_INT))
typedef highp_ivec1			ivec1;
#elif(defined(GLM_PRECISION_MEDIUMP_INT))
typedef mediump_ivec1		ivec1;
#elif(defined(GLM_PRECISION_LOWP_INT))
typedef lowp_ivec1			ivec1;
#else
typedef mediump_ivec1		ivec1;
#endif

#if(defined(GLM_PRECISION_HIGHP_UINT))
typedef highp_uvec1			uvec1;
#elif(defined(GLM_PRECISION_MEDIUMP_UINT))
typedef mediump_uvec1		uvec1;
#elif(defined(GLM_PRECISION_LOWP_UINT))
typedef lowp_uvec1			uvec1;
#else
typedef mediump_uvec1		uvec1;
#endif

}

#include "vec1.inl"

#endif

