
#pragma once

#include "../mat2x2.hpp"
#include "../mat2x3.hpp"
#include "../mat2x4.hpp"
#include "../mat3x2.hpp"
#include "../mat3x3.hpp"
#include "../mat3x4.hpp"
#include "../mat4x2.hpp"
#include "../mat4x3.hpp"
#include "../mat4x4.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_matrix_integer extension included")
#endif

namespace glm
{

typedef mat<2, 2, int, highp>				highp_imat2;

typedef mat<3, 3, int, highp>				highp_imat3;

typedef mat<4, 4, int, highp>				highp_imat4;

typedef mat<2, 2, int, highp>				highp_imat2x2;

typedef mat<2, 3, int, highp>				highp_imat2x3;

typedef mat<2, 4, int, highp>				highp_imat2x4;

typedef mat<3, 2, int, highp>				highp_imat3x2;

typedef mat<3, 3, int, highp>				highp_imat3x3;

typedef mat<3, 4, int, highp>				highp_imat3x4;

typedef mat<4, 2, int, highp>				highp_imat4x2;

typedef mat<4, 3, int, highp>				highp_imat4x3;

typedef mat<4, 4, int, highp>				highp_imat4x4;


typedef mat<2, 2, int, mediump>			mediump_imat2;

typedef mat<3, 3, int, mediump>			mediump_imat3;

typedef mat<4, 4, int, mediump>			mediump_imat4;


typedef mat<2, 2, int, mediump>			mediump_imat2x2;

typedef mat<2, 3, int, mediump>			mediump_imat2x3;

typedef mat<2, 4, int, mediump>			mediump_imat2x4;

typedef mat<3, 2, int, mediump>			mediump_imat3x2;

typedef mat<3, 3, int, mediump>			mediump_imat3x3;

typedef mat<3, 4, int, mediump>			mediump_imat3x4;

typedef mat<4, 2, int, mediump>			mediump_imat4x2;

typedef mat<4, 3, int, mediump>			mediump_imat4x3;

typedef mat<4, 4, int, mediump>			mediump_imat4x4;


typedef mat<2, 2, int, lowp>				lowp_imat2;

typedef mat<3, 3, int, lowp>				lowp_imat3;

typedef mat<4, 4, int, lowp>				lowp_imat4;


typedef mat<2, 2, int, lowp>				lowp_imat2x2;

typedef mat<2, 3, int, lowp>				lowp_imat2x3;

typedef mat<2, 4, int, lowp>				lowp_imat2x4;

typedef mat<3, 2, int, lowp>				lowp_imat3x2;

typedef mat<3, 3, int, lowp>				lowp_imat3x3;

typedef mat<3, 4, int, lowp>				lowp_imat3x4;

typedef mat<4, 2, int, lowp>				lowp_imat4x2;

typedef mat<4, 3, int, lowp>				lowp_imat4x3;

typedef mat<4, 4, int, lowp>				lowp_imat4x4;


typedef mat<2, 2, uint, highp>				highp_umat2;

typedef mat<3, 3, uint, highp>				highp_umat3;

typedef mat<4, 4, uint, highp>				highp_umat4;

typedef mat<2, 2, uint, highp>				highp_umat2x2;

typedef mat<2, 3, uint, highp>				highp_umat2x3;

typedef mat<2, 4, uint, highp>				highp_umat2x4;

typedef mat<3, 2, uint, highp>				highp_umat3x2;

typedef mat<3, 3, uint, highp>				highp_umat3x3;

typedef mat<3, 4, uint, highp>				highp_umat3x4;

typedef mat<4, 2, uint, highp>				highp_umat4x2;

typedef mat<4, 3, uint, highp>				highp_umat4x3;

typedef mat<4, 4, uint, highp>				highp_umat4x4;


typedef mat<2, 2, uint, mediump>			mediump_umat2;

typedef mat<3, 3, uint, mediump>			mediump_umat3;

typedef mat<4, 4, uint, mediump>			mediump_umat4;


typedef mat<2, 2, uint, mediump>			mediump_umat2x2;

typedef mat<2, 3, uint, mediump>			mediump_umat2x3;

typedef mat<2, 4, uint, mediump>			mediump_umat2x4;

typedef mat<3, 2, uint, mediump>			mediump_umat3x2;

typedef mat<3, 3, uint, mediump>			mediump_umat3x3;

typedef mat<3, 4, uint, mediump>			mediump_umat3x4;

typedef mat<4, 2, uint, mediump>			mediump_umat4x2;

typedef mat<4, 3, uint, mediump>			mediump_umat4x3;

typedef mat<4, 4, uint, mediump>			mediump_umat4x4;


typedef mat<2, 2, uint, lowp>				lowp_umat2;

typedef mat<3, 3, uint, lowp>				lowp_umat3;

typedef mat<4, 4, uint, lowp>				lowp_umat4;


typedef mat<2, 2, uint, lowp>				lowp_umat2x2;

typedef mat<2, 3, uint, lowp>				lowp_umat2x3;

typedef mat<2, 4, uint, lowp>				lowp_umat2x4;

typedef mat<3, 2, uint, lowp>				lowp_umat3x2;

typedef mat<3, 3, uint, lowp>				lowp_umat3x3;

typedef mat<3, 4, uint, lowp>				lowp_umat3x4;

typedef mat<4, 2, uint, lowp>				lowp_umat4x2;

typedef mat<4, 3, uint, lowp>				lowp_umat4x3;

typedef mat<4, 4, uint, lowp>				lowp_umat4x4;



typedef mat<2, 2, int, defaultp>				imat2;

typedef mat<3, 3, int, defaultp>				imat3;

typedef mat<4, 4, int, defaultp>				imat4;

typedef mat<2, 2, int, defaultp>				imat2x2;

typedef mat<2, 3, int, defaultp>				imat2x3;

typedef mat<2, 4, int, defaultp>				imat2x4;

typedef mat<3, 2, int, defaultp>				imat3x2;

typedef mat<3, 3, int, defaultp>				imat3x3;

typedef mat<3, 4, int, defaultp>				imat3x4;

typedef mat<4, 2, int, defaultp>				imat4x2;

typedef mat<4, 3, int, defaultp>				imat4x3;

typedef mat<4, 4, int, defaultp>				imat4x4;



typedef mat<2, 2, uint, defaultp>				umat2;

typedef mat<3, 3, uint, defaultp>				umat3;

typedef mat<4, 4, uint, defaultp>				umat4;

typedef mat<2, 2, uint, defaultp>				umat2x2;

typedef mat<2, 3, uint, defaultp>				umat2x3;

typedef mat<2, 4, uint, defaultp>				umat2x4;

typedef mat<3, 2, uint, defaultp>				umat3x2;

typedef mat<3, 3, uint, defaultp>				umat3x3;

typedef mat<3, 4, uint, defaultp>				umat3x4;

typedef mat<4, 2, uint, defaultp>				umat4x2;

typedef mat<4, 3, uint, defaultp>				umat4x3;

typedef mat<4, 4, uint, defaultp>				umat4x4;

}
