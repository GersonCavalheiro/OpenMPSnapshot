
#ifndef GLM_GTC_type_precision
#define GLM_GTC_type_precision GLM_VERSION

#include "../glm.hpp"
#include "../gtc/half_float.hpp"
#include "../gtc/quaternion.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTC_type_precision extension included")
#endif

namespace glm
{


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


typedef detail::tvec1<i8> i8vec1;

typedef detail::tvec2<i8> i8vec2;

typedef detail::tvec3<i8> i8vec3;

typedef detail::tvec4<i8> i8vec4;


typedef detail::tvec1<i16> i16vec1;

typedef detail::tvec2<i16> i16vec2;

typedef detail::tvec3<i16> i16vec3;

typedef detail::tvec4<i16> i16vec4;


typedef detail::tvec1<i32> i32vec1;

typedef detail::tvec2<i32> i32vec2;

typedef detail::tvec3<i32> i32vec3;

typedef detail::tvec4<i32> i32vec4;


typedef detail::tvec1<i64> i64vec1;

typedef detail::tvec2<i64> i64vec2;

typedef detail::tvec3<i64> i64vec3;

typedef detail::tvec4<i64> i64vec4;



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


typedef detail::tvec1<u8> u8vec1;

typedef detail::tvec2<u8> u8vec2;

typedef detail::tvec3<u8> u8vec3;

typedef detail::tvec4<u8> u8vec4;


typedef detail::tvec1<u16> u16vec1;

typedef detail::tvec2<u16> u16vec2;

typedef detail::tvec3<u16> u16vec3;

typedef detail::tvec4<u16> u16vec4;


typedef detail::tvec1<u32> u32vec1;

typedef detail::tvec2<u32> u32vec2;

typedef detail::tvec3<u32> u32vec3;

typedef detail::tvec4<u32> u32vec4;


typedef detail::tvec1<u64> u64vec1;

typedef detail::tvec2<u64> u64vec2;

typedef detail::tvec3<u64> u64vec3;

typedef detail::tvec4<u64> u64vec4;



typedef detail::float16 float16;

typedef detail::float32 float32;

typedef detail::float64 float64;


typedef detail::float16 float16_t;

typedef detail::float32 float32_t;

typedef detail::float64 float64_t;


typedef float16 f16;

typedef float32 f32;

typedef float64 f64;


typedef detail::tvec1<float> fvec1;

typedef detail::tvec2<float> fvec2;

typedef detail::tvec3<float> fvec3;

typedef detail::tvec4<float> fvec4;


typedef detail::tvec1<f16> f16vec1;

typedef detail::tvec2<f16> f16vec2;

typedef detail::tvec3<f16> f16vec3;

typedef detail::tvec4<f16> f16vec4;


typedef detail::tvec1<f32> f32vec1;

typedef detail::tvec2<f32> f32vec2;

typedef detail::tvec3<f32> f32vec3;

typedef detail::tvec4<f32> f32vec4;


typedef detail::tvec1<f64> f64vec1;

typedef detail::tvec2<f64> f64vec2;

typedef detail::tvec3<f64> f64vec3;

typedef detail::tvec4<f64> f64vec4;




typedef detail::tmat2x2<f32> fmat2;

typedef detail::tmat3x3<f32> fmat3;

typedef detail::tmat4x4<f32> fmat4;



typedef detail::tmat2x2<f32> fmat2x2;

typedef detail::tmat2x3<f32> fmat2x3;

typedef detail::tmat2x4<f32> fmat2x4;

typedef detail::tmat3x2<f32> fmat3x2;

typedef detail::tmat3x3<f32> fmat3x3;

typedef detail::tmat3x4<f32> fmat3x4;

typedef detail::tmat4x2<f32> fmat4x2;

typedef detail::tmat4x3<f32> fmat4x3;

typedef detail::tmat4x4<f32> fmat4x4;



typedef detail::tmat2x2<f16> f16mat2;

typedef detail::tmat3x3<f16> f16mat3;

typedef detail::tmat4x4<f16> f16mat4;



typedef detail::tmat2x2<f16> f16mat2x2;

typedef detail::tmat2x3<f16> f16mat2x3;

typedef detail::tmat2x4<f16> f16mat2x4;

typedef detail::tmat3x2<f16> f16mat3x2;

typedef detail::tmat3x3<f16> f16mat3x3;

typedef detail::tmat3x4<f16> f16mat3x4;

typedef detail::tmat4x2<f16> f16mat4x2;

typedef detail::tmat4x3<f16> f16mat4x3;

typedef detail::tmat4x4<f16> f16mat4x4;



typedef detail::tmat2x2<f32> f32mat2;

typedef detail::tmat3x3<f32> f32mat3;

typedef detail::tmat4x4<f32> f32mat4;



typedef detail::tmat2x2<f32> f32mat2x2;

typedef detail::tmat2x3<f32> f32mat2x3;

typedef detail::tmat2x4<f32> f32mat2x4;

typedef detail::tmat3x2<f32> f32mat3x2;

typedef detail::tmat3x3<f32> f32mat3x3;

typedef detail::tmat3x4<f32> f32mat3x4;

typedef detail::tmat4x2<f32> f32mat4x2;

typedef detail::tmat4x3<f32> f32mat4x3;

typedef detail::tmat4x4<f32> f32mat4x4;



typedef detail::tmat2x2<f64> f64mat2;

typedef detail::tmat3x3<f64> f64mat3;

typedef detail::tmat4x4<f64> f64mat4;



typedef detail::tmat2x2<f64> f64mat2x2;

typedef detail::tmat2x3<f64> f64mat2x3;

typedef detail::tmat2x4<f64> f64mat2x4;

typedef detail::tmat3x2<f64> f64mat3x2;

typedef detail::tmat3x3<f64> f64mat3x3;

typedef detail::tmat3x4<f64> f64mat3x4;

typedef detail::tmat4x2<f64> f64mat4x2;

typedef detail::tmat4x3<f64> f64mat4x3;

typedef detail::tmat4x4<f64> f64mat4x4;



typedef detail::tquat<f16> f16quat;

typedef detail::tquat<f32> f32quat;

typedef detail::tquat<f64> f64quat;

}

#include "type_precision.inl"

#endif
