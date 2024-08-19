#pragma once
#include "type_precision.hpp"
#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#pragma message("GLM: GLM_GTC_packing extension included")
#endif
namespace glm
{
GLM_FUNC_DECL uint8 packUnorm1x8(float v);
GLM_FUNC_DECL float unpackUnorm1x8(uint8 p);
GLM_FUNC_DECL uint16 packUnorm2x8(vec2 const & v);
GLM_FUNC_DECL vec2 unpackUnorm2x8(uint16 p);
GLM_FUNC_DECL uint8 packSnorm1x8(float s);
GLM_FUNC_DECL float unpackSnorm1x8(uint8 p);
GLM_FUNC_DECL uint16 packSnorm2x8(vec2 const & v);
GLM_FUNC_DECL vec2 unpackSnorm2x8(uint16 p);
GLM_FUNC_DECL uint16 packUnorm1x16(float v);
GLM_FUNC_DECL float unpackUnorm1x16(uint16 p);
GLM_FUNC_DECL uint64 packUnorm4x16(vec4 const & v);
GLM_FUNC_DECL vec4 unpackUnorm4x16(uint64 p);
GLM_FUNC_DECL uint16 packSnorm1x16(float v);
GLM_FUNC_DECL float unpackSnorm1x16(uint16 p);
GLM_FUNC_DECL uint64 packSnorm4x16(vec4 const & v);
GLM_FUNC_DECL vec4 unpackSnorm4x16(uint64 p);
GLM_FUNC_DECL uint16 packHalf1x16(float v);
GLM_FUNC_DECL float unpackHalf1x16(uint16 v);
GLM_FUNC_DECL uint64 packHalf4x16(vec4 const & v);
GLM_FUNC_DECL vec4 unpackHalf4x16(uint64 p);
GLM_FUNC_DECL uint32 packI3x10_1x2(ivec4 const & v);
GLM_FUNC_DECL ivec4 unpackI3x10_1x2(uint32 p);
GLM_FUNC_DECL uint32 packU3x10_1x2(uvec4 const & v);
GLM_FUNC_DECL uvec4 unpackU3x10_1x2(uint32 p);
GLM_FUNC_DECL uint32 packSnorm3x10_1x2(vec4 const & v);
GLM_FUNC_DECL vec4 unpackSnorm3x10_1x2(uint32 p);
GLM_FUNC_DECL uint32 packUnorm3x10_1x2(vec4 const & v);
GLM_FUNC_DECL vec4 unpackUnorm3x10_1x2(uint32 p);
GLM_FUNC_DECL uint32 packF2x11_1x10(vec3 const & v);
GLM_FUNC_DECL vec3 unpackF2x11_1x10(uint32 p);
}
#include "packing.inl"
