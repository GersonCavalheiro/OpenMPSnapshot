
#ifndef GLM_GTC_type_precision
#define GLM_GTC_type_precision

#include "../gtc/quaternion.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"
#include "../mat2x2.hpp"
#include "../mat2x3.hpp"
#include "../mat2x4.hpp"
#include "../mat3x2.hpp"
#include "../mat3x3.hpp"
#include "../mat3x4.hpp"
#include "../mat4x2.hpp"
#include "../mat4x3.hpp"
#include "../mat4x4.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_type_precision extension included")
#endif

namespace glm
{


typedef detail::int8 lowp_int8;

typedef detail::int16 lowp_int16;

typedef detail::int32 lowp_int32;

typedef detail::int64 lowp_int64;

typedef detail::int8 lowp_int8_t;

typedef detail::int16 lowp_int16_t;

typedef detail::int32 lowp_int32_t;

typedef detail::int64 lowp_int64_t;

typedef detail::int8 lowp_i8;

typedef detail::int16 lowp_i16;

typedef detail::int32 lowp_i32;

typedef detail::int64 lowp_i64;

typedef detail::int8 mediump_int8;

typedef detail::int16 mediump_int16;

typedef detail::int32 mediump_int32;

typedef detail::int64 mediump_int64;

typedef detail::int8 mediump_int8_t;

typedef detail::int16 mediump_int16_t;

typedef detail::int32 mediump_int32_t;

typedef detail::int64 mediump_int64_t;

typedef detail::int8 mediump_i8;

typedef detail::int16 mediump_i16;

typedef detail::int32 mediump_i32;

typedef detail::int64 mediump_i64;

typedef detail::int8 highp_int8;

typedef detail::int16 highp_int16;

typedef detail::int32 highp_int32;

typedef detail::int64 highp_int64;

typedef detail::int8 highp_int8_t;

typedef detail::int16 highp_int16_t;

typedef detail::int32 highp_int32_t;

typedef detail::int64 highp_int64_t;

typedef detail::int8 highp_i8;

typedef detail::int16 highp_i16;

typedef detail::int32 highp_i32;

typedef detail::int64 highp_i64;


typedef detail::int8 int8;

typedef detail::int16 int16;

typedef detail::int32 int32;

typedef detail::int64 int64;

typedef detail::int8 int8_t;

typedef detail::int16 int16_t;

typedef detail::int32 int32_t;

typedef detail::int64 int64_t;

typedef detail::int8 i8;

typedef detail::int16 i16;

typedef detail::int32 i32;

typedef detail::int64 i64;


typedef detail::tvec1<i8, defaultp> i8vec1;

typedef detail::tvec2<i8, defaultp> i8vec2;

typedef detail::tvec3<i8, defaultp> i8vec3;

typedef detail::tvec4<i8, defaultp> i8vec4;


typedef detail::tvec1<i16, defaultp> i16vec1;

typedef detail::tvec2<i16, defaultp> i16vec2;

typedef detail::tvec3<i16, defaultp> i16vec3;

typedef detail::tvec4<i16, defaultp> i16vec4;


typedef detail::tvec1<i32, defaultp> i32vec1;

typedef detail::tvec2<i32, defaultp> i32vec2;

typedef detail::tvec3<i32, defaultp> i32vec3;

typedef detail::tvec4<i32, defaultp> i32vec4;


typedef detail::tvec1<i64, defaultp> i64vec1;

typedef detail::tvec2<i64, defaultp> i64vec2;

typedef detail::tvec3<i64, defaultp> i64vec3;

typedef detail::tvec4<i64, defaultp> i64vec4;



typedef detail::uint8 lowp_uint8;

typedef detail::uint16 lowp_uint16;

typedef detail::uint32 lowp_uint32;

typedef detail::uint64 lowp_uint64;

typedef detail::uint8 lowp_uint8_t;

typedef detail::uint16 lowp_uint16_t;

typedef detail::uint32 lowp_uint32_t;

typedef detail::uint64 lowp_uint64_t;

typedef detail::uint8 lowp_u8;

typedef detail::uint16 lowp_u16;

typedef detail::uint32 lowp_u32;

typedef detail::uint64 lowp_u64;

typedef detail::uint8 mediump_uint8;

typedef detail::uint16 mediump_uint16;

typedef detail::uint32 mediump_uint32;

typedef detail::uint64 mediump_uint64;

typedef detail::uint8 mediump_uint8_t;

typedef detail::uint16 mediump_uint16_t;

typedef detail::uint32 mediump_uint32_t;

typedef detail::uint64 mediump_uint64_t;

typedef detail::uint8 mediump_u8;

typedef detail::uint16 mediump_u16;

typedef detail::uint32 mediump_u32;

typedef detail::uint64 mediump_u64;

typedef detail::uint8 highp_uint8;

typedef detail::uint16 highp_uint16;

typedef detail::uint32 highp_uint32;

typedef detail::uint64 highp_uint64;

typedef detail::uint8 highp_uint8_t;

typedef detail::uint16 highp_uint16_t;

typedef detail::uint32 highp_uint32_t;

typedef detail::uint64 highp_uint64_t;

typedef detail::uint8 highp_u8;

typedef detail::uint16 highp_u16;

typedef detail::uint32 highp_u32;

typedef detail::uint64 highp_u64;

typedef detail::uint8 uint8;

typedef detail::uint16 uint16;

typedef detail::uint32 uint32;

typedef detail::uint64 uint64;

typedef detail::uint8 uint8_t;

typedef detail::uint16 uint16_t;

typedef detail::uint32 uint32_t;

typedef detail::uint64 uint64_t;

typedef detail::uint8 u8;

typedef detail::uint16 u16;

typedef detail::uint32 u32;

typedef detail::uint64 u64;



typedef detail::tvec1<u8, defaultp> u8vec1;

typedef detail::tvec2<u8, defaultp> u8vec2;

typedef detail::tvec3<u8, defaultp> u8vec3;

typedef detail::tvec4<u8, defaultp> u8vec4;


typedef detail::tvec1<u16, defaultp> u16vec1;

typedef detail::tvec2<u16, defaultp> u16vec2;

typedef detail::tvec3<u16, defaultp> u16vec3;

typedef detail::tvec4<u16, defaultp> u16vec4;


typedef detail::tvec1<u32, defaultp> u32vec1;

typedef detail::tvec2<u32, defaultp> u32vec2;

typedef detail::tvec3<u32, defaultp> u32vec3;

typedef detail::tvec4<u32, defaultp> u32vec4;


typedef detail::tvec1<u64, defaultp> u64vec1;

typedef detail::tvec2<u64, defaultp> u64vec2;

typedef detail::tvec3<u64, defaultp> u64vec3;

typedef detail::tvec4<u64, defaultp> u64vec4;



typedef detail::float32 float32;

typedef detail::float64 float64;


typedef detail::float32 float32_t;

typedef detail::float64 float64_t;


typedef float32 f32;

typedef float64 f64;


typedef detail::tvec1<float, defaultp> fvec1;

typedef detail::tvec2<float, defaultp> fvec2;

typedef detail::tvec3<float, defaultp> fvec3;

typedef detail::tvec4<float, defaultp> fvec4;


typedef detail::tvec1<f32, defaultp> f32vec1;

typedef detail::tvec2<f32, defaultp> f32vec2;

typedef detail::tvec3<f32, defaultp> f32vec3;

typedef detail::tvec4<f32, defaultp> f32vec4;


typedef detail::tvec1<f64, defaultp> f64vec1;

typedef detail::tvec2<f64, defaultp> f64vec2;

typedef detail::tvec3<f64, defaultp> f64vec3;

typedef detail::tvec4<f64, defaultp> f64vec4;




typedef detail::tmat2x2<f32, defaultp> fmat2;

typedef detail::tmat3x3<f32, defaultp> fmat3;

typedef detail::tmat4x4<f32, defaultp> fmat4;



typedef detail::tmat2x2<f32, defaultp> fmat2x2;

typedef detail::tmat2x3<f32, defaultp> fmat2x3;

typedef detail::tmat2x4<f32, defaultp> fmat2x4;

typedef detail::tmat3x2<f32, defaultp> fmat3x2;

typedef detail::tmat3x3<f32, defaultp> fmat3x3;

typedef detail::tmat3x4<f32, defaultp> fmat3x4;

typedef detail::tmat4x2<f32, defaultp> fmat4x2;

typedef detail::tmat4x3<f32, defaultp> fmat4x3;

typedef detail::tmat4x4<f32, defaultp> fmat4x4;



typedef detail::tmat2x2<f32, defaultp> f32mat2;

typedef detail::tmat3x3<f32, defaultp> f32mat3;

typedef detail::tmat4x4<f32, defaultp> f32mat4;



typedef detail::tmat2x2<f32, defaultp> f32mat2x2;

typedef detail::tmat2x3<f32, defaultp> f32mat2x3;

typedef detail::tmat2x4<f32, defaultp> f32mat2x4;

typedef detail::tmat3x2<f32, defaultp> f32mat3x2;

typedef detail::tmat3x3<f32, defaultp> f32mat3x3;

typedef detail::tmat3x4<f32, defaultp> f32mat3x4;

typedef detail::tmat4x2<f32, defaultp> f32mat4x2;

typedef detail::tmat4x3<f32, defaultp> f32mat4x3;

typedef detail::tmat4x4<f32, defaultp> f32mat4x4;



typedef detail::tmat2x2<f64, defaultp> f64mat2;

typedef detail::tmat3x3<f64, defaultp> f64mat3;

typedef detail::tmat4x4<f64, defaultp> f64mat4;



typedef detail::tmat2x2<f64, defaultp> f64mat2x2;

typedef detail::tmat2x3<f64, defaultp> f64mat2x3;

typedef detail::tmat2x4<f64, defaultp> f64mat2x4;

typedef detail::tmat3x2<f64, defaultp> f64mat3x2;

typedef detail::tmat3x3<f64, defaultp> f64mat3x3;

typedef detail::tmat3x4<f64, defaultp> f64mat3x4;

typedef detail::tmat4x2<f64, defaultp> f64mat4x2;

typedef detail::tmat4x3<f64, defaultp> f64mat4x3;

typedef detail::tmat4x4<f64, defaultp> f64mat4x4;



typedef detail::tquat<f32, defaultp> f32quat;

typedef detail::tquat<f64, defaultp> f64quat;

}

#include "type_precision.inl"

#endif
